"""Post-hoc test of fixed grayscale/CLAHE soft fusion versus an equal-pass grayscale TTA control."""
from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
import torch
from model import NoduleUNet
from train import Pool, digest
from evaluate import measure

NAMES=('gray','contrast','gray_flip_average','gray_contrast_average','gray_closed','fusion_closed')


def closed(mask):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,kernel).astype(bool)


def bootstrap(delta, draws):
    return dict(mean_points=float(100*delta.mean()), interval_points=(100*np.quantile(delta[draws].mean(1),[.025,.975])).tolist(),
                improved=int((delta>0).sum()),worsened=int((delta<0).sum()),tied=int((delta==0).sum()))


def run(root):
    torch.set_num_threads(2)
    prior=root/'runs/source-fold0'; extra=root/'runs/ablation-fold0'
    pool=Pool(root/'local/data')
    base=json.loads((prior/'run.json').read_bytes())
    assert digest(root/'local/data/manifest.json')==base['manifest']
    assert digest(root/'train.py')==base['trainer'] and digest(root/'model.py')==base['model']
    networks={}; saved={}; receipts={}
    for arm,folder in [('gray',prior/'gray'),('contrast',extra/'contrast')]:
        receipt=json.loads((folder/'evaluation.json').read_bytes())
        assert digest(folder/'checkpoint.pt')==receipt['checkpoint_sha256']
        assert digest(folder/'predictions.npz')==receipt['predictions_sha256']
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        net=NoduleUNet();net.load_state_dict(checkpoint['model']);net.eval()
        networks[arm]=net; receipts[arm]=receipt
        del checkpoint
        with np.load(folder/'predictions.npz') as z:
            np.testing.assert_array_equal(pool.val,z['ids'])
            saved[arm]=np.unpackbits(z['bits'],axis=-1)[:,:,:pool.n].astype(bool)
    assert receipts['gray']['initial_sha256']==receipts['contrast']['initial_sha256']
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    results=[]; discrepancies={k:[0,0,0.] for k in networks}
    with torch.inference_mode():
        for start in range(0,len(pool.val),4):
            ids=pool.val[start:start+4]
            x,_,_=pool.batch(ids,'edge','cpu')
            gray=((x[:,[0,0,0]]/255)-mean)/std
            contrast=((x[:,[0,1,0]]/255)-mean)/std
            pg=networks['gray'](gray).sigmoid()
            pc=networks['contrast'](contrast).sigmoid()
            pf=networks['gray'](gray.flip([-1])).sigmoid().flip([-1])
            output=[(prob[:,0]>=.5).numpy() for prob in (pg,pc,(pg+pf)/2,(pg+pc)/2)]
            for j,i in enumerate(ids):
                r=pool.rows[i];left,top,w,h=r['box']
                sl=np.s_[top:top+h,left:left+w]
                truth=pool.gt[i][sl].astype(bool)
                masks=[o[j][sl] for o in output];masks.extend([closed(masks[0]),closed(masks[3])])
                scores={name:measure(pred,truth) for name,pred in zip(NAMES,masks,strict=True)}
                for k,arm in enumerate(('gray','contrast')):
                    expected=saved[arm][start+j][sl]
                    n=int(np.count_nonzero(expected!=masks[k]))
                    if n:
                        stats=discrepancies[arm];stats[0]+=1;stats[1]+=n
                        stats[2]=max(stats[2],abs(measure(expected,truth)['dice']-scores[arm]['dice']))
                results.append(dict(i=int(i),scores=scores))
    rng=np.random.default_rng(241125);draws=rng.integers(len(results),size=(5000,len(results)))
    def vector(arm,metric):return np.array([r['scores'][arm][metric] for r in results])
    summary={}
    for arm in NAMES:
        summary[arm]={m:float(vector(arm,m).mean()) for m in ('dice','surface_dice','hd95')}
        summary[arm]['misses']=int(vector(arm,'miss').sum())
    comparisons={}
    for arm,control in [('contrast','gray'),('gray_flip_average','gray'),('gray_contrast_average','gray'),
                        ('gray_contrast_average','contrast'),('gray_contrast_average','gray_flip_average'),
                        ('gray_closed','gray'),('fusion_closed','gray_contrast_average')]:
        comparisons[arm+' minus '+control]=bootstrap(vector(arm,'dice')-vector(control,'dice'),draws)
    individual=root/'runs/ablation-fold0/fusion-audit-paired.json'
    individual.write_text(json.dumps(results,allow_nan=False)+'\n')
    result=dict(images=len(results),device='cpu',methods=summary,comparisons=comparisons,
                cpu_vs_mps_discrepancy_note='Per arm: images with any difference, total different pixels, maximum absolute image Dice difference inside unpadded field.',
                cpu_vs_mps=discrepancies,checkpoints={a:v['checkpoint_sha256'] for a,v in receipts.items()},
                per_image_sha256=digest(individual),script_sha256=digest(__file__))
    (root/'results/fusion-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run(Path(__file__).resolve().parent)

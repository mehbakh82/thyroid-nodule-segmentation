"""Post-hoc, CPU-only sensitivity of fold-0 boundary comparisons to label decoding and tolerance."""
import argparse
import json
from pathlib import Path
from review_records import save_key

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from evaluate import sha


def score(prediction, target):
    if not target.any():
        raise ValueError('Empty target under an alternate decoder')
    tp=np.count_nonzero(prediction & target)
    dice=2*tp/(prediction.sum()+target.sum())
    if not prediction.any():
        return float(dice),np.zeros(3)
    pred_edge=prediction ^ ndi.binary_erosion(prediction)
    ref_edge=target ^ ndi.binary_erosion(target)
    distances=np.concatenate((ndi.distance_transform_edt(~ref_edge)[pred_edge],
                              ndi.distance_transform_edt(~pred_edge)[ref_edge]))
    return float(dice),np.array([(distances<=t).mean() for t in (1,2,3)])


def run(root, labels):
    data=root/'local/data'
    original=root/'runs/source-fold0'
    out=original/'boundary-audit'
    out.mkdir(exist_ok=True)
    summary=json.loads((root/'results/summary.json').read_bytes())
    manifest=json.loads((data/'manifest.json').read_bytes())
    assert sha(data/'manifest.json')==summary['run']['manifest']
    assert sha(original/'paired.json')==summary['paired_sha256']
    baseline={r['i']:r for r in json.loads((original/'paired.json').read_bytes())}
    rows=[r for r in manifest['rows'] if r['role']=='validation']
    inputs={}
    for arm in ('gray','edge'):
        path=original/arm/'predictions.npz'
        assert sha(path)==summary['receipts'][arm]['predictions_sha256']
        with np.load(path) as z:
            np.testing.assert_array_equal(z['ids'],[r['i'] for r in rows])
            inputs[arm]=np.unpackbits(z['bits'],axis=-1)[:,:,:manifest['size']].astype(bool)
    metrics=[]; labels_meta=[]
    for j,r in enumerate(rows):
        path=labels/r['file']
        assert sha(path)==r['mask_sha256']
        with Image.open(path) as f: raw=np.asarray(f.convert('L'))
        assert raw.max()==255
        x,y,w,h=r['box'];values=[]
        for cut in (127,254):
            target=np.array(Image.fromarray((raw>cut).astype('uint8')).resize((w,h),Image.Resampling.NEAREST),bool)
            scores=[]
            for arm in ('gray','edge'):
                dice,surface=score(inputs[arm][j,y:y+h,x:x+w],target)
                if cut==127:
                    np.testing.assert_allclose([dice,surface[1]],
                        [baseline[r['i']][arm]['dice'],baseline[r['i']][arm]['surface_dice']],atol=1e-10)
                scores.append([dice,*surface.tolist()])
            values.append(scores)
        labels_meta.append(r['i'])
        metrics.append(values)
    m=np.asarray(metrics)  # image, decoder, arm, (region Dice, boundary at 1/2/3)
    delta=m[:,:,1,1:]-m[:,:,0,1:]
    local=(m[:,0,:,0]>=.50).all(axis=1)
    both_local=(m[:,:,:,0]>=.50).all(axis=(1,2))
    plus=local & (delta[:,0,1]>=.10)
    minus=local & (delta[:,0,1]<=-.10)
    stable_plus=both_local & (delta.min(axis=(1,2))>=.10)
    stable_minus=both_local & (delta.max(axis=(1,2))<=-.10)
    same_sign_plus=plus & both_local & (delta.min(axis=(1,2))>0)
    same_sign_minus=minus & both_local & (delta.max(axis=(1,2))<0)
    def count(a):return int(a.sum())
    result=dict(n=len(rows),primary_both_local=count(local),both_decoders_both_local=count(both_local),
                primary_plus=count(plus),primary_minus=count(minus),robust_plus=count(stable_plus),
                robust_minus=count(stable_minus),positive_sign_only_or_better=count(same_sign_plus),
                negative_sign_only_or_better=count(same_sign_minus),conditions={})
    for k,decoder in enumerate(('above127','only255')):
        cond={'region_gray':float(m[:,k,0,0].mean()),'region_edge':float(m[:,k,1,0].mean()),
              'delta_region':float(np.mean(m[:,k,1,0]-m[:,k,0,0]))}
        for t in (1,2,3):
            cond[f'surface_delta_at_{t}px']=float(delta[:,k,t-1].mean())
        result['conditions'][decoder]=cond
    assert (len(rows),count(plus),count(minus))==(576,67,70)
    (root/'results/boundary-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    # Keep detailed sampling information out of the reader packet.
    previously_shown={int(i) for _,i in json.loads((original/'panel-cases.json').read_bytes())}
    groups={'persistent improvement':stable_plus,'persistent worsening':stable_minus,
            'sensitive to scoring choice':(plus|minus)&~(stable_plus|stable_minus),
            'little change':both_local & (np.abs(delta).max(axis=(1,2))<.025),
            'complete miss by both':(m[:,0,:,0]==0).all(axis=1)}
    rng=np.random.default_rng(260121)
    selected=[]
    for name,mask in groups.items():
        pool=[int(i) for i,b in zip(labels_meta,mask) if b and i not in previously_shown]
        n=3 if name.startswith('persistent') else 2
        assert len(pool)>=n,(name,len(pool))
        for i in rng.choice(sorted(pool),n,replace=False):selected.append(dict(i=int(i),stratum=name))
    rng.shuffle(selected)
    assignments=['gray','edge']*(len(selected)//2)
    rng.shuffle(assignments)
    for j,(r,arm) in enumerate(zip(selected,assignments)):
        r.update(case=j+1,A=arm)
    assert len({r['i'] for r in selected})==len(selected)
    save_key(out,selected)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trainval_masks',type=Path)
    run(Path(__file__).resolve().parent,p.parse_args().trainval_masks)

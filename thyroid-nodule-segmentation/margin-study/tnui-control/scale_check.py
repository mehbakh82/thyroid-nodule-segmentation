"""Fixed-scale follow-up. Compare whole-field views on a common evaluation grid."""
import argparse,json,io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from codec_control import ROOT,file_hash,get_binary,NoduleUNet,score,mean_ci,Path,hashlib

def predict(net,im,size):
    w,h=[max(1,round(d*size/max(im.size))) for d in im.size]
    x,y=(size-w)//2,(size-h)//2
    a=np.zeros((size,size),dtype='f4')
    a[y:y+h,x:x+w]=np.asarray(im.resize((w,h),Image.Resampling.BILINEAR),dtype='f4')/255
    z=(torch.from_numpy(a)[None,None,:,:]-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    p=(net(z).sigmoid()+net(z.flip([-1])).sigmoid().flip([-1]))/2
    return p[:,:,y:y+h,x:x+w]

def run(data):
    torch.set_num_threads(2)
    folder=Path(__file__).resolve().parent
    base=json.loads((folder/'results.json').read_bytes())
    receipt=json.loads((data/'sample.json').read_bytes());assert base['sample_sha256']==file_hash(data/'sample.json')
    assert base['selected']==base['eligible']==48
    previous=json.loads((data/'per-image-scores.json').read_bytes())
    cp=ROOT/'runs/source-fold0/gray/checkpoint.pt';assert base['checkpoint_sha256']==file_hash(cp)
    net=NoduleUNet();ck=torch.load(cp,map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);del ck;net.eval()
    rows=[]
    with torch.inference_mode():
        for item,ref in zip(receipt['rows'],previous,strict=True):
            files={}
            for k in ('images','masks'):
                b=(data/k/item['name']).read_bytes()
                assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item[k]
                with Image.open(io.BytesIO(b)) as f:files[k]=f.convert('L')
            im=files['images'];raw=get_binary(np.array(files['masks']));assert raw is not None
            p224=predict(net,im,224)
            p448=F.interpolate(predict(net,im,448),size=p224.shape[-2:],mode='bilinear',align_corners=False)
            target=np.array(Image.fromarray(raw.astype('uint8')).resize(tuple(p224.shape[-2:][::-1]),Image.Resampling.NEAREST),bool)
            case={'area_fraction':float(raw.mean())}
            for name,pred in [('224',p224),('448',p448)]:
                d,s=score(pred[0,0].numpy()>=.5,target)
                case[name]={'dice':float(d),'surface2':float(s[1])}
            assert np.isclose(case['224']['dice'],ref['png']['dice'])
            rows.append(case)
    draws=np.random.default_rng(231126).integers(len(rows),size=(5000,len(rows)))
    levels={name:{k:float(np.mean([r[name][k] for r in rows])) for k in ('dice','surface2')} for name in ('224','448')}
    for name,v in levels.items():v['zero_overlap']=sum(r[name]['dice']==0 for r in rows)
    changes={k:mean_ci([100*(r['448'][k]-r['224'][k]) for r in rows],draws) for k in ('dice','surface2')}
    result=dict(images=len(rows),median_native_area_fraction=float(np.median([r['area_fraction'] for r in rows])),
                levels=levels,paired_change_points=changes,gains=sum(r['448']['dice']>r['224']['dice'] for r in rows),
                losses=sum(r['448']['dice']<r['224']['dice'] for r in rows),baseline_score_receipt=file_hash(folder/'results.json'))
    (folder/'scale-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sample',type=Path);run(p.parse_args().sample)

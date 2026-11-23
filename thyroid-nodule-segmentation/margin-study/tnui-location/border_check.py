"""Image-only horizontal display-border diagnostic on the existing development sample."""
import json
import numpy as np
import torch
from pilot import ROOT,HERE,load_pair,native_prediction,NoduleUNet,file_hash,measure


def scan_bounds(a):
    h,w=a.shape
    start,stop=w//20,w-w//20
    active=np.mean(a[:,start:stop]>8,axis=1)>=.10
    edges=np.diff(np.r_[False,active,False].astype(int))
    runs=[(l,r) for l,r in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if r-l>=5]
    return (int(runs[0][0]),int(runs[-1][1])) if runs else (0,h)


def run():
    torch.set_num_threads(2)
    data=ROOT/'local/tnui-location'
    result=json.loads((HERE/'results.json').read_bytes())
    assert file_hash(data/'sample.json')==result['sample_sha256']
    assert file_hash(data/'per-image.json')==result['per_image_sha256']
    old=json.loads((data/'per-image.json').read_bytes());rows=[]
    table={r['name']:r for r in json.loads((data/'sample.json').read_bytes())['rows']}
    cp=ROOT/'runs/source-fold0/gray/checkpoint.pt'
    assert file_hash(cp)==result['checkpoint_sha256']
    net=NoduleUNet();ck=torch.load(cp,map_location='cpu',weights_only=True)
    net.load_state_dict(ck['model']);del ck;net.eval()
    with torch.inference_mode():
        for case in old:
            name=case['name'];(im,mask),_=load_pair(data,table[name])
            whole=native_prediction(net,im,(0,0,*im.size))
            original=measure(whole,mask)
            assert np.isclose(original['dice'],case['scores']['whole']['dice'])
            top,bottom=scan_bounds(np.asarray(im));keep=np.zeros_like(mask);keep[top:bottom,:]=True
            filtered=measure(whole&keep,mask)
            rows.append(dict(name=name,top=top,bottom=bottom,height=im.height,
                             target=int(mask.sum()),target_outside=int((mask&~keep).sum()),fp=original['fp'],
                             fp_removed=original['fp']-filtered['fp'],tp_removed=int((mask&whole&~keep).sum()),
                             dice=filtered['dice'],surface2=filtered['surface_dice'],miss=filtered['miss'],
                             raw_dice=original['dice'],central_dice=case['scores']['center']['dice']))
    n=len(rows)
    draws=np.random.default_rng(231128).integers(n,size=(5000,n))
    def delta(ref):
        d=np.array([100*(v['dice']-v[ref]) for v in rows])
        return dict(mean=float(d.mean()),ci=np.quantile(d[draws].mean(1),[.025,.975]).tolist(),
                    improved=int((d>0).sum()),worse=int((d<0).sum()))
    def total(key):return sum(v[key] for v in rows)
    summary=dict(n=n,any_removed_rows=sum(r['top']>0 or r['bottom']<r['height'] for r in rows),
                 affected_references=sum(r['target_outside']>0 for r in rows),target_pixels=total('target'),
                 target_pixels_outside=total('target_outside'),true_positives_removed=total('tp_removed'),
                 whole_fp=total('fp'),false_positives_removed=total('fp_removed'),
                 fraction_whole_fp_removed=total('fp_removed')/total('fp'),
                 mean_dice=total('dice')/n,mean_surface_dice=total('surface2')/n,misses=total('miss'),
                 vs_whole=delta('raw_dice'),vs_central=delta('central_dice'),
                 primary_per_image_sha256=result['per_image_sha256'])
    (data/'border-per-image.json').write_text(json.dumps(rows,allow_nan=False)+'\n')
    (HERE/'border-results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(summary)

if __name__=='__main__':run()

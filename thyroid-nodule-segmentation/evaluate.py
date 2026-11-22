"""Paired image-level report from saved, fixed-threshold validation predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from review import comparison

METRICS=('dice','iou','surface_dice','hd95','fp_per_target','fn_per_target')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def measure(pred,gt):
    n=int(gt.sum())
    if not n:raise ValueError('positive reference required for this study')
    tp=int((pred & gt).sum());fp=int(pred.sum())-tp;fn=n-tp
    if pred.any():
        bp=pred ^ ndi.binary_erosion(pred)
        bg=gt ^ ndi.binary_erosion(gt)
        dtg=ndi.distance_transform_edt(~bg)[bp]
        dtp=ndi.distance_transform_edt(~bp)[bg]
        hd=float(np.percentile(np.concatenate([dtg,dtp]),95))
        sd=float((np.count_nonzero(dtg<=2)+np.count_nonzero(dtp<=2))/(len(dtg)+len(dtp)))
    else:
        hd=float(np.hypot(*gt.shape));sd=0.
    return dict(dice=2*tp/(2*tp+fp+fn),iou=tp/(tp+fp+fn),surface_dice=sd,hd95=hd,fp=fp,fn=fn,
                fp_per_target=fp/n,fn_per_target=fn/n,miss=tp==0,empty=not bool(pred.any()))


def summarize(cases):
    if not cases:return dict(n=0)
    result={'n':len(cases)}
    for arm in ('gray','edge'):
        result[arm]={m:float(np.mean([c[arm][m] for c in cases])) for m in METRICS}
        result[arm].update({m:sum(c[arm][m] for c in cases) for m in ('miss','empty')})
    d=np.array([c['delta'] for c in cases]); rng=np.random.default_rng(241122)
    avg=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
    result.update(mean_delta=float(d.mean()),median_delta=float(np.median(d)),wins=int((d>0).sum()),losses=int((d<0).sum()),
                  ties=int((d==0).sum()),conditional_image_interval=np.quantile(avg,[.025,.975]).tolist(),
                  rescue=sum(c['rescue'] for c in cases),harm=sum(c['harm'] for c in cases),
                  recovered_any=sum(c['gray']['miss'] and not c['edge']['miss'] for c in cases),
                  lost_any=sum(c['edge']['miss'] and not c['gray']['miss'] for c in cases),
                  both_miss=sum(c['gray']['miss'] and c['edge']['miss'] for c in cases),
                  extra_foreground_worse=sum(c['edge']['fp']>c['gray']['fp'] and c['delta']<0 for c in cases))
    return result


def evaluate(data,run,markers):
    meta=json.loads((data/'manifest.json').read_text())
    spec=json.loads((run/'run.json').read_text())
    assert spec['manifest']==sha(data/'manifest.json')
    for fname,value in meta['arrays'].items():assert sha(data/fname)==value
    target=np.load(data/'masks.npy',mmap_mode='r')
    preds={};receipts={}
    ids=[r['i'] for r in meta['rows'] if r['role']=='validation']
    for arm in ('gray','edge'):
        r=json.loads((run/arm/'evaluation.json').read_text());file=run/arm/'predictions.npz'
        assert r['predictions_sha256']==sha(file) and r['epochs']==spec['epochs']
        with np.load(file) as z:
            assert np.array_equal(z['ids'],ids)
            preds[arm]=np.unpackbits(z['bits'],axis=-1)[:,:,:meta['size']].astype(bool)
        receipts[arm]=r
    assert receipts['gray']['initial_sha256']==receipts['edge']['initial_sha256']
    assert receipts['gray']['run_key']==receipts['edge']['run_key']
    labels=json.loads(markers.read_text())['labels']
    cases=[]
    for j,i in enumerate(ids):
        r=meta['rows'][i];x,y,w,h=r['box']
        rows,cols=slice(y,y+h),slice(x,x+w)
        gt=target[i,rows,cols].astype(bool)
        a,b=[measure(preds[arm][j,rows,cols],gt) for arm in ('gray','edge')]
        cases.append(dict(i=i,file=r['file'],gray=a,edge=b,delta=b['dice']-a['dice'],
                          small=r['area']<=meta['subgroup']['area_q25'],
                          weak=r['margin'] is not None and r['margin']<=meta['subgroup']['margin_q25'],
                          proxy_available=r['margin'] is not None,marker=labels.get(str(i)),
                          rescue=a['miss'] and b['dice']>=.5,harm=b['miss'] and a['dice']>=.5))
    pair=run/'paired.json';pair.write_text(json.dumps(cases,allow_nan=False)+'\n')
    rules={'small':lambda r:r['small'],'other_size':lambda r:not r['small'],'weak':lambda r:r['weak'],
           'other_margin':lambda r:r['proxy_available'] and not r['weak'],
           'marked_review':lambda r:r['marker']=='clear','no_obvious_marks_review':lambda r:r['marker']=='no_obvious',
           'uncertain_review':lambda r:r['marker']=='uncertain'}
    result=dict(run=spec,receipts=receipts,overall=summarize(cases),
                groups={g:summarize([c for c in cases if rule(c)]) for g,rule in rules.items()},
                bounds=meta['subgroup'],margin_proxy_missing=sum(not c['proxy_available'] for c in cases),
                paired_sha256=sha(pair),marker_review_sha256=sha(markers))
    (run/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    comparison(data,run)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--markers',type=Path,required=True);a=p.parse_args()
    print(json.dumps(evaluate(a.data,a.run,a.markers)['overall'],indent=2))

"""Replication of the display filter with a fixed-position control on fresh source images."""
import argparse
from collections import Counter
import json

import numpy as np
import torch

from border_check import scan_bounds
from pilot import ROOT,HERE,Path,load_pair,native_prediction,NoduleUNet,file_hash,measure,identity,Image


def evaluate(tn):
    torch.set_num_threads(2)
    data=ROOT/'local/tnui-border-repeat'
    manifest=json.loads((data/'sample.json').read_bytes())
    previous=ROOT/'local/tnui-location';first=ROOT/'local/tnui-control'
    assert manifest['location_sample_sha256']==file_hash(previous/'sample.json') and manifest['prior_sha256']==file_hash(first/'sample.json')
    assert manifest['salt']=='tnui-display-repeat-v1/' and manifest['partition']=='train' and len(manifest['rows'])==48
    bounds_file=previous/'border-per-image.json';old=json.loads(bounds_file.read_bytes())
    assert len(old)==96
    frac=np.median([[r['top']/r['height'],r['bottom']/r['height']] for r in old],axis=0)
    seen=set();previous_names=set()
    for p in (first,previous):
        for r in json.loads((p/'sample.json').read_bytes())['rows']:
            previous_names.add(r['name'])
            with Image.open(p/'images'/r['name']) as im:seen.add(identity(im))
    names={r['name'] for r in manifest['rows']}
    assert len(names)==48 and len(previous_names)==144 and names.isdisjoint(previous_names)
    for r in json.loads((ROOT/'local/data/manifest.json').read_bytes())['rows']:
        src=tn/'trainval-image'/r['file'];assert file_hash(src)==r['image_sha256']
        with Image.open(src) as im:seen.add(identity(im))
    cp=ROOT/'runs/source-fold0/gray/checkpoint.pt';expected=json.loads((cp.parent/'evaluation.json').read_bytes())['checkpoint_sha256']
    assert file_hash(cp)==expected
    net=NoduleUNet();ck=torch.load(cp,map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);del ck;net.eval()
    rows=[];ex=Counter()
    with torch.inference_mode():
        for r in manifest['rows']:
            pair,reason=load_pair(data,r)
            if reason:ex[reason]+=1;continue
            im,gt=pair;ident=identity(im)
            if ident in seen:ex['duplicate source pixels']+=1;continue
            seen.add(ident)
            whole=native_prediction(net,im,(0,0,*im.size))
            row={'name':r['name'],'target':int(gt.sum())}
            for name,(lo,hi) in [('whole',(0,im.height)),('adaptive',scan_bounds(np.asarray(im))),('fixed',np.rint(frac*im.height).astype(int))]:
                keep=np.zeros_like(gt);keep[lo:hi,:]=True
                m=measure(whole&keep,gt)
                row[name]={k:m[k] for k in ('dice','surface_dice','miss','fp','fn')}
                row[name]['target_outside']=int((gt&~keep).sum())
            rows.append(row)
    draws=np.random.default_rng(231129).integers(len(rows),size=(5000,len(rows)));methods={};gains={}
    for arm in ('whole','adaptive','fixed'):
        metrics=[r[arm] for r in rows]
        methods[arm]=dict(mean_dice=float(np.mean([m['dice'] for m in metrics])),mean_surface_dice=float(np.mean([m['surface_dice'] for m in metrics])),
            misses=sum(m['miss'] for m in metrics),fp=sum(m['fp'] for m in metrics),fn=sum(m['fn'] for m in metrics),
            target_pixels_outside=sum(m['target_outside'] for m in metrics),affected_images=sum(m['target_outside']>0 for m in metrics))
    for arm,control in [('adaptive','whole'),('fixed','whole'),('adaptive','fixed')]:
        delta=np.array([100*(r[arm]['dice']-r[control]['dice']) for r in rows])
        gains[arm+'_minus_'+control]=dict(mean=float(delta.mean()),ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),improved=int((delta>0).sum()),worse=int((delta<0).sum()))
    path=data/'per-image.json';path.write_text(json.dumps(rows,allow_nan=False)+'\n')
    total=sum(r['target'] for r in rows)
    result=dict(n=len(rows),selected=48,exclusions=dict(ex),methods=methods,paired_dice_points=gains,
        median_bounds_from_previous_96=frac.tolist(),pooled_reference_retention=1-methods['adaptive']['target_pixels_outside']/total,
        pooled_fp_reduction=1-methods['adaptive']['fp']/methods['whole']['fp'],reference_pixels=total,
        removed_true_positives=methods['adaptive']['fn']-methods['whole']['fn'],sample_sha256=file_hash(data/'sample.json'),
        old_bounds_sha256=file_hash(bounds_file),protocol_sha256=file_hash(HERE/'BORDER.md'),per_image_sha256=file_hash(path))
    result['replicated']=bool(gains['adaptive_minus_whole']['ci'][0]>0 and result['pooled_reference_retention']>=.99)
    (HERE/'border-repeat.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['replicated'],gains,methods)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('tn3k',type=Path)
    evaluate(p.parse_args().tn3k)

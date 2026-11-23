"""Compare whole-field and fixed-size crop localization on held-out-from-prior-probe training images."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F

from fetch import blob, REV, REPO
CONTROL=Path(__file__).resolve().parents[1]/'tnui-control'
sys.path.insert(0,str(CONTROL))
from scale_check import predict
from codec_control import ROOT,identity,file_hash,get_binary
from model import NoduleUNet
from evaluate import measure

HERE=Path(__file__).resolve().parent
ARMS=('whole','center','oracle','automatic','clipped_whole')


def centered_box(size,xy):
    width,height=size
    side=max(1,round(min(size)/2))
    left=min(max(0,round(xy[0]-(side-1)/2)),width-side)
    top=min(max(0,round(xy[1]-(side-1)/2)),height-side)
    return (left,top,left+side,top+side)


def centroid(binary):
    y,x=ndi.center_of_mass(binary)
    return float(x),float(y)


def automatic_center(whole):
    labels,n=ndi.label(whole,structure=np.ones((3,3)))
    if not n:
        h,w=whole.shape
        return ((w-1)/2,(h-1)/2),True
    largest=np.bincount(labels.ravel())[1:].argmax()+1
    return centroid(labels==largest),False


def native_prediction(net,image,box):
    left,top,right,bottom=box
    p=predict(net,image.crop(box),224)
    scaled=F.interpolate(p,size=(bottom-top,right-left),mode='bilinear',align_corners=False)
    h,w=image.height,image.width
    result=np.zeros((h,w),bool)
    result[top:bottom,left:right]=scaled[0,0].numpy()>=.5
    return result


def windows(mask,whole):
    h,w=mask.shape
    ac,fallback=automatic_center(whole)
    return dict(center=centered_box((w,h),((w-1)/2,(h-1)/2)),
                oracle=centered_box((w,h),centroid(mask)),
                automatic=centered_box((w,h),ac)),fallback


def localized_predictions(net,image,mask):
    whole=native_prediction(net,image,(0,0,image.width,image.height))
    boxes,fallback=windows(mask,whole)
    predictions={'whole':whole}
    coverage={}
    for arm,box in boxes.items():
        left,top,right,bottom=box
        coverage[arm]=float(mask[top:bottom,left:right].sum()/mask.sum())
        predictions[arm]=native_prediction(net,image,box)
    left,top,right,bottom=boxes['oracle']
    clipped=np.zeros_like(whole);clipped[top:bottom,left:right]=whole[top:bottom,left:right]
    predictions['clipped_whole']=clipped
    return predictions,coverage,fallback


def load_pair(data,row):
    arrays=[]
    for k in ('images','masks'):
        raw=(data/k/row['name']).read_bytes()
        assert blob(raw)==row[k]
        with Image.open(io.BytesIO(raw)) as im:
            if k=='masks' and (im.format!='PNG' or im.mode not in ('1','L','P')):
                return None,'unsupported mask'
            arrays.append(im.convert('L'))
    image,label=arrays
    if image.size!=label.size:return None,'mismatched size'
    mask=get_binary(np.asarray(label))
    if mask is None:return None,'nonbinary or empty mask'
    return (image,mask),None


def summarize(rows):
    n=len(rows)
    if not n:raise ValueError('No eligible images')
    draws=np.random.default_rng(231127).integers(n,size=(5000,n))
    summary={};comparisons={}
    def values(a,k):return np.array([r['scores'][a][k] for r in rows])
    for a in ARMS:
        summary[a]={k:float(np.mean(values(a,k))) for k in ('dice','sensitivity','fp_per_target','surface_dice','hd95')}
        summary[a]['misses']=int(sum(values(a,'miss')))
        summary[a]['at_least_half_dice']=int(sum(values(a,'dice')>=.5))
    for a,b in [('oracle','center'),('oracle','whole'),('center','whole'),('automatic','whole'),
                ('automatic','center'),('oracle','automatic'),('clipped_whole','whole'),('oracle','clipped_whole')]:
        d=100*(values(a,'dice')-values(b,'dice'))
        comparisons[a+'_minus_'+b]=dict(mean_points=float(d.mean()),ci_points=np.quantile(d[draws].mean(1),[.025,.975]).tolist(),
                                       median_points=float(np.median(d)),improved=int((d>0).sum()),worse=int((d<0).sum()))
    coverage={}
    for a in ('center','oracle','automatic'):
        cov=np.array([r['coverage'][a] for r in rows])
        coverage[a]={'mean':float(cov.mean()),'fully_contains_target':int((cov==1).sum()),'excludes_entire_target':int((cov==0).sum())}
    primary=comparisons['oracle_minus_center']
    advance=primary['mean_points']>=5 and primary['ci_points'][0]>0 and summary['oracle']['dice']>summary['whole']['dice']
    return dict(n=n,methods=summary,comparisons=comparisons,coverage=coverage,advance_localization_opportunity=bool(advance),
                auto_center_fallback=sum(r['fallback'] for r in rows))


def run(data,prior,tn):
    torch.set_num_threads(2)
    checkpoint=ROOT/'runs/source-fold0/gray/checkpoint.pt'
    receipt=json.loads((checkpoint.parent/'evaluation.json').read_bytes())
    assert file_hash(checkpoint)==receipt['checkpoint_sha256']
    net=NoduleUNet();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['model']);del state;net.eval()
    r=json.loads((data/'sample.json').read_bytes());old=json.loads((prior/'sample.json').read_bytes())
    assert r['prior_sha256']==file_hash(prior/'sample.json')
    assert r['revision']==old['revision']==REV and r['repo']==REPO and r['partition']=='train'
    names={x['name'] for x in r['rows']}
    assert len(names)==len(r['rows'])==96 and names.isdisjoint(x['name'] for x in old['rows'])
    hashes=set()
    for case in old['rows']:
        with Image.open(prior/'images'/case['name']) as f:hashes.add(identity(f))
    meta=json.loads((ROOT/'local/data/manifest.json').read_bytes())
    for item in meta['rows']:
        f=tn/'trainval-image'/item['file']
        assert file_hash(f)==item['image_sha256']
        with Image.open(f) as im:hashes.add(identity(im))
    skip=Counter();new=set();rows=[]
    with torch.inference_mode():
        for row in r['rows']:
            pair,reason=load_pair(data,row)
            if reason:skip[reason]+=1;continue
            image,mask=pair;digest=identity(image)
            if digest in hashes:skip['matches earlier development image']+=1;continue
            if digest in new:skip['duplicate selected image']+=1;continue
            new.add(digest)
            preds,cover,fallback=localized_predictions(net,image,mask)
            scores={}
            for a,p in preds.items():
                m=measure(p,mask)
                scores[a]={k:m[k] for k in ('dice','surface_dice','hd95','fp_per_target','miss')}
                scores[a]['sensitivity']=1-m['fn_per_target']
            rows.append(dict(name=row['name'],scores=scores,coverage=cover,fallback=fallback,
                             native_size=image.size,area_fraction=float(mask.mean())))
            if len(rows)%24==0:print('scored',len(rows),flush=True)
    path=data/'per-image.json'
    path.write_text(json.dumps(rows,allow_nan=False)+'\n')
    summary=summarize(rows)|dict(selected=96,exclusions=dict(skip),area_fraction_median=float(np.median([r['area_fraction'] for r in rows])),
        sizes=dict(Counter('x'.join(map(str,r['native_size'])) for r in rows)),checkpoint_sha256=receipt['checkpoint_sha256'],
        sample_sha256=file_hash(data/'sample.json'),per_image_sha256=file_hash(path),protocol_sha256=file_hash(HERE/'PROTOCOL.md'),
        code_sha256=file_hash(Path(__file__)))
    (HERE/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(summary['comparisons']['oracle_minus_center'],summary['advance_localization_opportunity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('sample','prior','tn3k'):p.add_argument(k,type=Path)
    a=p.parse_args();run(a.sample,a.prior,a.tn3k)

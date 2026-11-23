"""Training-partition format control. Uses a frozen TN3K prediction at all reference settings."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model import NoduleUNet
from boundary_audit import score
from fetch_sample import REV,SALT

def file_hash(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity(im):
    return hashlib.sha256(str(im.size).encode()+im.convert('L').tobytes()).hexdigest()

def get_binary(raw):
    vals=np.unique(raw)
    if len(vals)!=2 or vals[0]!=0:return None
    return raw==vals[1]

def transcode(mask,qt):
    stream=io.BytesIO()
    Image.fromarray(mask.astype('uint8')*255).save(stream,format='JPEG',qtables=[list(qt)])
    with Image.open(io.BytesIO(stream.getvalue())) as im:
        assert tuple(im.quantization[0])==qt
        return np.asarray(im.convert('L'))

def mean_ci(vals,draw):
    a=np.asarray(vals)
    return {'mean':float(a.mean()),'interval':np.quantile(a[draw].mean(axis=1),[.025,.975]).tolist()}

def run(data,tn):
    torch.set_num_threads(2)
    meta=json.loads((ROOT/'local/data/manifest.json').read_bytes())
    counts=Counter();existing=set()
    for row in meta['rows']:
        mask=tn/'trainval-mask'/row['file']
        image=tn/'trainval-image'/row['file']
        assert file_hash(mask)==row['mask_sha256'] and file_hash(image)==row['image_sha256']
        with Image.open(mask) as im:counts[tuple(im.quantization[0])]+=1
        with Image.open(image) as im:existing.add(identity(im))
    qt,n=counts.most_common(1)[0]
    checkpoint=ROOT/'runs/source-fold0/gray/checkpoint.pt'
    receipt=json.loads((checkpoint.parent/'evaluation.json').read_text())
    assert file_hash(checkpoint)==receipt['checkpoint_sha256']
    model=NoduleUNet();ck=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);del ck;model.eval()
    manifest=json.loads((data/'sample.json').read_bytes())
    assert manifest['revision']==REV and manifest['partition']=='train'
    assert len({v['name'] for v in manifest['rows']})==len(manifest['rows'])==48
    records=[];exceptions=Counter();seen=set()
    for r in manifest['rows']:
        blobs={}
        for k in ('images','masks'):
            b=(data/k/r['name']).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r[k]
            blobs[k]=b
        with Image.open(io.BytesIO(blobs['images'])) as im,Image.open(io.BytesIO(blobs['masks'])) as ref:
            im=im.convert('L')
            if im.size!=ref.size:exceptions['size mismatch']+=1;continue
            ident=identity(im)
            if ident in existing:exceptions['matches TN3K development']+=1;continue
            if ident in seen:exceptions['duplicate TNUI image']+=1;continue
            if ref.mode not in ('1','L','P') or ref.format!='PNG':exceptions['unsupported mask format']+=1;continue
            original=get_binary(np.asarray(ref.convert('L')))
            if original is None:exceptions['nonbinary or empty release mask']+=1;continue
            dims=tuple(max(1,round(224*d/max(im.size))) for d in im.size)
            w,h=dims;x=(224-w)//2;y=(224-h)//2
            a=np.zeros((224,224),dtype='f4')
            a[y:y+h,x:x+w]=np.asarray(im.resize(dims,Image.Resampling.BILINEAR),dtype='f4')/255
        gray=torch.from_numpy(a)[None,None,:,:]
        z=(gray-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            p=(model(z).sigmoid()+model(z.flip([-1])).sigmoid().flip([-1]))/2
        base=(p[0,0,y:y+h,x:x+w].numpy()>=.5);shrunk=ndi.binary_erosion(base)
        encoded=transcode(original,qt)
        labeldefs={'png':original,'jpeg_halfmax':encoded>127,'jpeg_only255':encoded==255}
        scaled={k:np.asarray(Image.fromarray(v.astype('uint8')).resize(dims,Image.Resampling.NEAREST),bool) for k,v in labeldefs.items()}
        if not all(v.any() for v in scaled.values()):exceptions['reference vanishes at 224']+=1;continue
        rec={}
        for key,mask in scaled.items():
            d,sd=score(base,mask);e,se=score(shrunk,mask)
            target=labeldefs[key]
            rec[key]={'dice':float(d),'surface2':float(sd[1]),'shrink_delta_points':float(100*(se[1]-sd[1])),
                      'native_reference_agreement':float(2*np.count_nonzero(target&original)/(target.sum()+original.sum()))}
        seen.add(ident);records.append(rec)
    if not records:raise ValueError('No eligible mask pairs')
    draws=np.random.default_rng(231126).integers(len(records),size=(5000,len(records)))
    effects={k:np.array([r[k]['shrink_delta_points'] for r in records]) for k in labeldefs}
    overall={k:{'gray_flip_region_dice':float(np.mean([r[k]['dice'] for r in records])),
                'gray_flip_surface2':float(np.mean([r[k]['surface2'] for r in records])),
                'erosion_change_points':mean_ci(effects[k],draws),
                'median_native_mask_agreement':float(np.median([r[k]['native_reference_agreement'] for r in records]))} for k in labeldefs}
    result=dict(dataset=manifest['repo'],revision=REV,salt=SALT,split='train',selected=48,eligible=len(records),
                exclusions=dict(exceptions),sample_sha256=file_hash(data/'sample.json'),checkpoint_sha256=file_hash(checkpoint),
                tn3k_most_common_luminance_table=list(qt),table_occurrences=n,tn3k_development_masks=sum(counts.values()),
                overall=overall,paired_change_in_erosion_effect_strict_minus_png_points=mean_ci(effects['jpeg_only255']-effects['png'],draws),
                per_image_opposite_signs=int(np.count_nonzero(effects['png']*effects['jpeg_only255']<0)))
    (Path(__file__).parent/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (data/'per-image-scores.json').write_text(json.dumps(records,allow_nan=False)+'\n')
    print(len(records),exceptions,overall,result['paired_change_in_erosion_effect_strict_minus_png_points'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sample',type=Path);p.add_argument('tn3k',type=Path);a=p.parse_args()
    run(a.sample,a.tn3k)

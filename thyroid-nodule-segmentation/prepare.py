"""Prepare only the official TN3K development pool. The split indices use numeric filename order."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import find_contours

SIZE=224
SEED=241121


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,obj):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def margin_proxy(im,mask):
    """Offline subgroup descriptor. Positive means stronger gradient on boundary than nearby tissue."""
    gradient=ndimage.gaussian_gradient_magnitude(im.astype('f4')/255,1)
    signed=ndimage.distance_transform_edt(mask)-ndimage.distance_transform_edt(~mask)
    gy,gx=np.gradient(ndimage.gaussian_filter(signed,1))
    samples=[]
    for c in find_contours(mask,.5):
        c=c.T
        def interp(image,pos):return ndimage.map_coordinates(image,pos,order=1,mode='nearest')
        v=np.array([interp(gy,c),interp(gx,c)])
        lengths=np.linalg.norm(v,axis=0)
        v/=np.maximum(lengths,.00001)
        a,b=c+3*v,c-3*v
        valid=(lengths>.1)&(np.min(np.minimum(a,b),axis=0)>=2)&np.all(
            np.maximum(a,b)<np.asarray(im.shape)[:,None]-2,axis=0)
        valid&=(interp(signed,a)>0)&(interp(signed,b)<0)
        if valid.any():samples.append(np.array([interp(gradient,p[:,valid]) for p in (c,a,b)]))
    if not samples or sum(x.shape[1] for x in samples)<20:return None
    g,i,o=np.median(np.concatenate(samples,axis=1),axis=1)
    return float(g-max(i,o))


def prepare(root,out):
    out.mkdir(parents=True,exist_ok=True)
    files=sorted((root/'trainval-image').glob('*.jpg'),key=lambda p:int(p.stem))
    splitfile=root/'tn3k-trainval-fold0.json'
    split=json.loads(splitfile.read_bytes())
    train,val=map(set,(split['train'],split['val']))
    assert len(files)==2879 and len(train)==2303 and len(val)==576
    assert not train&val and train|val==set(range(len(files)))
    images=np.zeros((len(files),SIZE,SIZE),dtype='uint8')
    masks=np.zeros_like(images);rows=[]; fingerprints=[];thumbs=[]; hashkeys=[]
    for i,f in enumerate(files):
        mf=root/'trainval-mask'/f.name
        with Image.open(f) as im, Image.open(mf) as mask:
            im=im.convert('L')
            raw=np.asarray(mask.convert('L'))
            assert im.size==mask.size and raw.max()==255
            gt=raw>127
            long_side=max(im.size)
            w,h=[max(1,round(d*SIZE/long_side)) for d in im.size]
            x,y=(SIZE-w)//2,(SIZE-h)//2
            image=np.array(im.resize((w,h),Image.Resampling.BILINEAR))
            label=np.array(Image.fromarray(gt.astype('uint8')).resize((w,h),Image.Resampling.NEAREST))
            assert label.any()
            images[i,y:y+h,x:x+w]=image;masks[i,y:y+h,x:x+w]=label
            small=np.array(im.resize((9,8),Image.Resampling.BILINEAR))
            hashkeys.append(int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),'big'))
            thumb=np.array(im.resize((64,64),Image.Resampling.BILINEAR),dtype='f4').ravel()
            thumb-=thumb.mean();thumbs.append(thumb/max(np.linalg.norm(thumb),1e-6))
            fingerprints.append(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
            rows.append(dict(i=i,file=f.name,role='train' if i in train else 'validation',box=[x,y,w,h],
                             area=float(gt.mean()),margin=margin_proxy(image,label.astype(bool)),
                             image_sha256=sha(f),mask_sha256=sha(mf)))
    cross=[]
    sorted_val=sorted(val)
    for i in sorted(train):
        for j in sorted_val:
            if fingerprints[i]==fingerprints[j] or ((hashkeys[i]^hashkeys[j]).bit_count()<=4 and thumbs[i]@thumbs[j]>=.995):
                rows[i]['role']='excluded';cross.append([i,j])
    tr=[r for r in rows if r['role']=='train']
    subgroup=dict(area_q25=float(np.quantile([r['area'] for r in tr],.25)),
                  margin_q25=float(np.quantile([r['margin'] for r in tr if r['margin'] is not None],.25)))
    np.save(out/'images.npy',images);np.save(out/'masks.npy',masks)
    save(out/'manifest.json',dict(rows=rows,subgroup=subgroup,excluded_cross_split=cross,split_sha256=sha(splitfile),
        arrays={f:sha(out/f) for f in ['images.npy','masks.npy']},size=SIZE))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args()
    prepare(args.root,args.out)

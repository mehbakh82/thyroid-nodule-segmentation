"""Development-only descriptive probe of image gradients along a supplied contour."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import find_contours

def measure(image,mask):
    if image.shape!=mask.shape: return 'dimension mismatch'
    if not mask.any(): return 'empty mask'
    h,w=image.shape
    size=tuple(max(1,round(d*256/max(h,w))) for d in (w,h))
    im=np.asarray(Image.fromarray(image).resize(size,Image.Resampling.BILINEAR),dtype='f4')/255
    m=np.asarray(Image.fromarray(mask.astype('uint8')).resize(size,Image.Resampling.NEAREST),dtype=bool)
    if not m.any() or m.all():return 'degenerate after resize'
    # Raw CT-style intensity calibration has no analogue here: this is displayed B-mode grayscale.
    grad=ndi.gaussian_gradient_magnitude(im,1)
    signed=ndi.distance_transform_edt(m)-ndi.distance_transform_edt(~m)
    dy,dx=np.gradient(ndi.gaussian_filter(signed,1))
    probes=[]
    for boundary in find_contours(m,.5):
        if len(boundary)<2:continue
        c=boundary.T
        def sample(arr,pos):return ndi.map_coordinates(arr,pos,order=1,mode='nearest')
        norm=np.array([sample(dy,c),sample(dx,c)])
        length=np.linalg.norm(norm,axis=0)
        v=norm/np.maximum(length,1e-6)
        a=c+3*v;b=c-3*v
        ok=(length>.1)&(np.min(np.minimum(a,b),axis=0)>=2)&np.all(np.maximum(a,b)<np.array(m.shape)[:,None]-2,axis=0)
        ok&=(sample(signed,a)>0)&(sample(signed,b)<0)
        if ok.any():probes.append(np.stack([sample(grad,p[:,ok]) for p in (c,a,b)]))
    if not probes or sum(x.shape[1] for x in probes)<20:return 'too few valid locations'
    g,i,o=np.concatenate(probes,axis=1)
    return (float(np.median(g)-max(np.median(i),np.median(o))),float(np.mean((g>i)&(g>o))),len(g))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('tn3k',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args()
    rows=[];ex=Counter()
    for p in sorted((args.tn3k/'trainval-image').glob('*.jpg')):
        im=np.asarray(Image.open(p).convert('L'))
        raw=np.asarray(Image.open(args.tn3k/'trainval-mask'/p.name).convert('L'))
        v=measure(im,raw.astype(float)>raw.max()/2)
        if isinstance(v,str):ex[v]+=1
        else:rows.append(v)
    a=np.array(rows)
    args.out.write_text(json.dumps(dict(split='trainval only',eligible_images=len(rows),exclusions=ex,
        boundary_median_exceeds_both_control_medians=int(sum(a[:,0]>0)),
        image_quantiles_fraction_boundary_points_stronger_than_both_controls=np.quantile(a[:,1],[.1,.5,.9]).tolist(),
        protocol='long side 256, isotropic resize, sigma 1 gradient, parallel samples at plus/minus 3 pixels',mask='per-mask half-maximum'),indent=2)+'\n')

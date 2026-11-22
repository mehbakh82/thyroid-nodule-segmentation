"""Seven marked TN3K development scans, fixed small-disk inpainting and equal-area spatial controls."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from model import NoduleUNet
from train import Pool,channels,digest


def disks(shape,points,radius=4):
    yy,xx=np.indices(shape)
    return np.logical_or.reduce([(xx-x)**2+(yy-y)**2<=radius**2 for x,y in points])


def probes(shape,points,reference):
    real=disks(shape,points)
    options=[]
    for x,y in [(12,0),(-12,0),(0,12),(0,-12),(9,9),(-9,9),(9,-9),(-9,-9),
                (18,0),(-18,0),(0,18),(0,-18)]:
        shifted=disks(shape,[(a+x,b+y) for a,b in points])
        if real.sum()!=shifted.sum() or np.any(real&shifted):continue
        diff=abs(int(reference[real].sum())-int(reference[shifted].sum()))
        options.append((diff,shifted))
    if not options:raise ValueError('No equal-area control in field')
    # Label composition is used only to choose diagnostic control locations.
    # No predictions or confidence scores participate in selecting edits.
    options.sort(key=lambda x:x[0])
    return real,[v for _,v in options[:3]]


def run(root):
    torch.set_num_threads(2)
    pool=Pool(root/'local/data')
    order=json.loads((root/'local/blind/blind-order.json').read_bytes())
    loc=json.loads((root/'marker_locations.json').read_bytes())['points']
    rows=[]
    for arm in ('gray','contrast'):
        folder=root/'runs'/('source-fold0' if arm=='gray' else 'ablation-fold0')/arm
        receipt=json.loads((folder/'evaluation.json').read_bytes())
        assert digest(folder/'checkpoint.pt')==receipt['checkpoint_sha256']
        net=NoduleUNet();checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(checkpoint['model']);net.eval();del checkpoint
        for slot,points in loc.items():
            i=order[int(slot)-1];r=pool.rows[i];left,top,w,h=r['box']
            assert r['role']=='validation'
            sl=np.s_[top:top+h,left:left+w]
            im=pool.im[i][sl];gt=pool.gt[i][sl].astype(bool)
            real,controls=probes(gt.shape,[(x-left,y-top) for x,y in points],gt)
            inp=[im]+[cv2.inpaint(im,m.astype('uint8'),3,cv2.INPAINT_TELEA) for m in [real]+controls]
            batch=np.zeros((len(inp),3,pool.n,pool.n),dtype='f4')
            for j,v in enumerate(inp):
                c=channels(v,'edge' if arm=='contrast' else 'gray')
                if arm=='contrast':c[2]=v
                batch[j,:,top:top+h,left:left+w]=c
            mean=torch.tensor([.485,.456,.406])[None,:,None,None]
            std=torch.tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode():pred=net((torch.from_numpy(batch)/255-mean)/std)[:,0,sl[0],sl[1]].numpy()>=0
            scores=2*(pred & gt).sum(axis=(1,2))/(pred.sum(axis=(1,2))+gt.sum())
            rows.append(dict(arm=arm,slot=int(slot),points=len(points),controls=len(controls),
                edited_pixels=int(real.sum()),original=float(scores[0]),real=float(scores[1]),sham=float(scores[2:].mean()),
                difference=float(scores[1]-scores[2:].mean()),
                mean_background_fraction_mismatch=float(np.mean([abs(gt[real].mean()-gt[m].mean()) for m in controls]))))
    summary={}
    for arm in ('gray','contrast'):
        arr=[v for v in rows if v['arm']==arm]
        summary[arm]={key:float(np.mean([v[key] for v in arr])) for key in ('original','real','sham','difference','mean_background_fraction_mismatch')}
    result=dict(cases=len(loc),arms=summary,rows=rows,
                coordinates_sha256=digest(root/'marker_locations.json'),
                reader_order_sha256=digest(root/'local/blind/blind-order.json'))
    (root/'results/mark-intervention.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(summary)


if __name__=='__main__':run(Path(__file__).resolve().parent)

"""Matched, fixed-budget development runs. Launch one writer for a run directory."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from model import NoduleUNet

EPOCHS=12
BATCH=8
SEED=241121
ARMS=('gray','edge')
cv2.setNumThreads(1)


def digest(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def augment(im,mask,epoch,index):
    rng=np.random.default_rng([SEED,epoch,index])
    h,w=im.shape
    if rng.random()<.5:
        im,mask=np.fliplr(im),np.fliplr(mask)
    affine=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),rng.uniform(-10,10),rng.uniform(.93,1.07))
    affine[:,2]+=rng.uniform(-.035,.035,2)*[w,h]
    im=cv2.warpAffine(im,affine,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    mask=cv2.warpAffine(mask,affine,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
    lut=np.clip((np.arange(256)/255)**rng.uniform(.8,1.25)*255*rng.uniform(.9,1.1)+rng.uniform(-8,8),0,255)
    return cv2.LUT(im,np.rint(lut).astype('uint8')),mask


def channels(im,arm):
    if arm=='gray':return np.stack([im,im,im])
    clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(im)
    gx=cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
    mag=cv2.magnitude(gx,gy)
    grad=np.rint(mag*255/max(float(mag.max()),1.)).clip(0,255).astype('uint8')
    return np.stack([im,clahe,grad])


class Pool:
    def __init__(self,folder):
        self.manifest=json.loads((folder/'manifest.json').read_text())
        for f,sha in self.manifest['arrays'].items():
            if digest(folder/f)!=sha:raise ValueError('modified data cache: '+f)
        self.im=np.load(folder/'images.npy',mmap_mode='r')
        self.gt=np.load(folder/'masks.npy',mmap_mode='r')
        self.rows=self.manifest['rows']
        self.train=np.array([r['i'] for r in self.rows if r['role']=='train'])
        self.val=np.array([r['i'] for r in self.rows if r['role']=='validation'])
        assert len(self.val)==576 and not set(self.train)&set(self.val)
        self.n=self.manifest['size']

    def batch(self,ids,arm,device,epoch=None):
        x=np.zeros((len(ids),3,self.n,self.n),dtype='uint8')
        y=np.zeros((len(ids),1,self.n,self.n),dtype='uint8')
        valid=np.zeros_like(y)
        for k,i in enumerate(ids):
            left,top,width,height=self.rows[i]['box']
            a,b=slice(top,top+height),slice(left,left+width)
            im=self.im[i,a,b]; mask=self.gt[i,a,b]
            if epoch is not None:im,mask=augment(im,mask,epoch,int(i))
            x[k,:,a,b]=channels(im,arm)
            y[k,0,a,b]=mask; valid[k,0,a,b]=1
        return tuple(torch.from_numpy(t).to(device=device,dtype=torch.float32) for t in (x,y,valid))


def loss_fn(logits,y,valid):
    bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*valid).sum()/valid.sum()
    pred=logits.sigmoid()*valid
    dice=(2*(pred*y).sum((1,2,3))+1)/(pred.sum((1,2,3))+y.sum((1,2,3))+1)
    return .5*(bce+1-dice.mean())


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_arm(pool,arm,folder,weights,device,run_key):
    out=folder/arm
    out.mkdir(parents=True,exist_ok=True)
    final=out/'evaluation.json'
    if final.exists():
        saved=json.loads(final.read_text())
        if saved['run_key']!=run_key or digest(out/'predictions.npz')!=saved['predictions_sha256']:
            raise ValueError('completed arm does not match this run')
        return
    torch.manual_seed(SEED)
    model=NoduleUNet(weights);initial=state_hash(model);model.to(device)
    decoder=list(model.decoder.parameters())+list(model.logits.parameters())
    decoder_ids={id(p) for p in decoder}
    encoder=[p for p in model.parameters() if id(p) not in decoder_ids]
    opt=torch.optim.AdamW([{'params':encoder,'lr':.0002},{'params':decoder,'lr':.001}],weight_decay=.0001,foreach=False)
    means=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    stds=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    cp=out/'checkpoint.pt'; history=[]
    if cp.exists():
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        if ck['run_key']!=run_key or ck['arm']!=arm or ck['initial']!=initial:raise ValueError('incompatible checkpoint')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);history=ck['history']
        del ck
    for epoch in range(len(history),EPOCHS):
        model.train()
        start=time.monotonic()
        order=np.random.default_rng([SEED,epoch]).permutation(pool.train)
        summed=torch.zeros((),device=device)
        for j in range(0,len(order),BATCH):
            progress=(epoch+j/len(order))/EPOCHS
            factor=.1+.9*(1+math.cos(math.pi*progress))/2
            for group,lr in zip(opt.param_groups,(.0002,.001)):group['lr']=lr*factor
            ids=order[j:j+BATCH]
            x,y,v=pool.batch(ids,arm,device,epoch)
            opt.zero_grad(set_to_none=True)
            z=model((x/255-means)/stds)
            loss=loss_fn(z,y,v)
            loss.backward();opt.step()
            summed+=loss.detach()*len(ids)
            if j==0 and epoch==0:print(arm,'first batch loss',float(loss.detach().cpu()),flush=True)
        avg=float(summed.cpu())/len(order)
        if not math.isfinite(avg):raise ValueError('nonfinite epoch loss')
        history.append(dict(epoch=epoch+1,loss=avg,seconds=time.monotonic()-start))
        tmp=cp.with_suffix('.tmp')
        torch.save(dict(run_key=run_key,arm=arm,initial=initial,model=model.state_dict(),
                        opt=opt.state_dict(),history=history),tmp);tmp.replace(cp)
        write(out/'history.json',history)
        print(arm,history[-1],flush=True)
    model.eval(); masks=[]
    with torch.inference_mode():
        for batch in np.array_split(pool.val,math.ceil(len(pool.val)/BATCH)):
            x,_,v=pool.batch(batch,arm,device)
            pred=(model((x/255-means)/stds)>=0)&(v>0)
            masks.append(np.packbits(pred[:,0].cpu().numpy(),axis=2))
    np.savez_compressed(out/'predictions.npz',ids=pool.val,bits=np.concatenate(masks))
    write(final,dict(run_key=run_key,initial_sha256=initial,arm=arm,parameters=sum(p.numel() for p in model.parameters()),
                     epochs=len(history),predictions_sha256=digest(out/'predictions.npz'),checkpoint_sha256=digest(cp)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',choices=['mps','cuda','cpu'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.device=='cuda':
        mem=torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.mem_get_info()[0]<6*1024**3:raise RuntimeError('insufficient available CUDA memory')
        torch.cuda.set_per_process_memory_fraction(min(1.,20_000_000_000/mem),0)
    weight_hash=digest(args.weights)
    if not weight_hash.startswith('b627a593'):raise ValueError('incorrect pretrained weights')
    data=Pool(args.data);args.run.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    spec=dict(device=args.device,torch=str(torch.__version__),opencv=cv2.__version__,python=platform.python_version(),
              manifest=digest(args.data/'manifest.json'),weights=weight_hash,train=len(data.train),val=len(data.val),
              model=digest(source/'model.py'),trainer=digest(__file__),epochs=EPOCHS,batch=BATCH,seed=SEED)
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if (args.run/'run.json').exists() and json.loads((args.run/'run.json').read_text())!=spec:raise ValueError('run mismatch')
    write(args.run/'run.json',spec)
    for arm in ARMS:run_arm(data,arm,args.run,args.weights,args.device,key)


if __name__=='__main__':main()

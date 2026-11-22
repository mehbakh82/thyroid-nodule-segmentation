"""Complete the fold-0 input ablation on the same Mac and verify the two new fits."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

import train
from model import NoduleUNet


class MacMonitor:
    states=('nominal','fair','serious','critical')

    def __init__(self,path):
        self.path=path
        self.next_poll=0
        self.last_status=None
        ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        pointer=ctypes.c_void_p
        for routine in (objc.objc_getClass,objc.sel_registerName):
            routine.argtypes=[ctypes.c_char_p]
            routine.restype=pointer
        send=ctypes.CFUNCTYPE(pointer,pointer,pointer)(('objc_msgSend',objc))
        self.info=send(objc.objc_getClass(b'NSProcessInfo'),objc.sel_registerName(b'processInfo'))
        self.selector=objc.sel_registerName(b'thermalState')
        self.read=ctypes.CFUNCTYPE(ctypes.c_long,pointer,pointer)(('objc_msgSend',objc))

    def check(self,force=False):
        if not force and time.monotonic()<self.next_poll:return
        while True:
            level=self.read(self.info,self.selector)
            if not 0<=level<len(self.states):raise RuntimeError('Unknown thermal status')
            power=subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True)
            ac='AC Power' in power
            status=(self.states[level],ac)
            record=dict(time=time.time(),thermal=status[0],ac_power=ac,
                        driver_bytes=torch.mps.driver_allocated_memory())
            with self.path.open('a') as f:f.write(json.dumps(record)+'\n')
            if status!=self.last_status:
                print('Mac monitor:',record,flush=True)
                self.last_status=status
            if level<2 and ac:break
            torch.mps.synchronize()
            print('Pausing GPU work until thermal pressure eases and AC power is available.',flush=True)
            time.sleep(30)
        self.next_poll=time.monotonic()+15


class AblationPool(train.Pool):
    """Replace just the missing channel with grayscale before the original normalization."""
    monitor=None

    def batch(self,ids,arm,device,epoch=None):
        if arm not in ('contrast','gradient'):raise ValueError(arm)
        if self.monitor:self.monitor.check()
        x,y,v=super().batch(ids,'edge',device,epoch)
        replace={'contrast':2,'gradient':1}[arm]
        x[:,replace].copy_(x[:,0])
        return x,y,v


def verify_checkpoint(pool,folder,arm,original):
    receipt=json.loads((folder/'evaluation.json').read_bytes())
    assert receipt['arm']==arm and receipt['initial_sha256']==original['initial_sha256']
    assert receipt['epochs']==train.EPOCHS and receipt['parameters']==original['parameters']
    assert receipt['checkpoint_sha256']==train.digest(folder/'checkpoint.pt')
    assert receipt['predictions_sha256']==train.digest(folder/'predictions.npz')
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['run_key']==receipt['run_key']
    model=NoduleUNet()
    model.load_state_dict(ck['model'])
    del ck
    model.to('mps').eval()
    means=torch.tensor([.485,.456,.406],device='mps')[None,:,None,None]
    stds=torch.tensor([.229,.224,.225],device='mps')[None,:,None,None]
    with np.load(folder/'predictions.npz') as f:
        ids,expected=f['ids'],f['bits']
    np.testing.assert_array_equal(pool.val,ids)
    with torch.inference_mode():
        for j in range(0,len(ids),train.BATCH):
            x,_,v=pool.batch(ids[j:j+train.BATCH],arm,'mps')
            pred=(model((x/255-means)/stds)>=0)&(v>0)
            np.testing.assert_array_equal(np.packbits(pred[:,0].cpu().numpy(),axis=2),expected[j:j+train.BATCH])
    return len(ids)


def main(folder):
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU unavailable')
    torch.mps.set_per_process_memory_fraction(1.)
    data=Path('local/data'); weights=Path('local/resnet34-b627a593.pth')
    prior=Path('runs/source-fold0')
    pool=AblationPool(data)
    spec=json.loads((prior/'run.json').read_text())
    observed=dict(device='mps',model=train.digest('model.py'),trainer=train.digest('train.py'),
                  torch=str(torch.__version__),opencv=cv2.__version__,python=platform.python_version(),
                  weights=train.digest(weights),manifest=train.digest(data/'manifest.json'),
                  train=len(pool.train),val=len(pool.val),seed=train.SEED,epochs=train.EPOCHS,batch=train.BATCH)
    if spec!=observed:raise RuntimeError('Original development training recipe does not match')
    # Sanity-check the identical field, augmentation and channel positions on a training batch.
    ids=pool.train[:3]
    full,y,v=train.Pool.batch(pool,ids,'edge','cpu',0)
    for arm,keep,replace in [('contrast',1,2),('gradient',2,1)]:
        x,yy,vv=pool.batch(ids,arm,'cpu',0)
        assert torch.equal(y,yy) and torch.equal(v,vv)
        assert torch.equal(x[:,0],full[:,0]) and torch.equal(x[:,keep],full[:,keep])
        assert torch.equal(x[:,replace],full[:,0]) and not torch.equal(full[:,replace],full[:,0])
    pool.monitor=MacMonitor(folder/'thermal.jsonl');pool.monitor.check(force=True)
    observed.update(ablation=train.digest(__file__),baseline_run=train.digest(prior/'run.json'))
    key=hashlib.sha256(json.dumps(observed,sort_keys=True).encode()).hexdigest()
    if (folder/'run.json').exists() and json.loads((folder/'run.json').read_bytes())!=observed:
        raise RuntimeError('Different ablation already present at destination')
    train.write(folder/'run.json',observed)
    initial=json.loads((prior/'gray/evaluation.json').read_bytes())
    replay={}
    for arm in ('contrast','gradient'):
        train.run_arm(pool,arm,folder,weights,'mps',key)
        replay[arm]=verify_checkpoint(pool,folder/arm,arm,initial)
        train.write(folder/'replay.json',replay)
        torch.mps.empty_cache()
    pool.monitor.check(force=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('runs/ablation-fold0'))
    args=p.parse_args();args.run.mkdir(exist_ok=True,parents=True)
    with (args.run/'.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.run)

"""Recreate two-view baseline probabilities on CPU; tie the cache to the original fold."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import NoduleUNet
from train import Pool,digest,write

if __name__=='__main__':
    torch.set_num_threads(2)
    data=ROOT/'local/data'
    original=ROOT/'runs/source-fold0'
    record=json.loads((original/'gray/evaluation.json').read_bytes())
    checkpoint=original/'gray/checkpoint.pt'
    assert digest(checkpoint)==record['checkpoint_sha256']
    assert digest(data/'manifest.json')==json.loads((original/'run.json').read_bytes())['manifest']
    pool=Pool(data)
    dest=ROOT/'local/margin-study';dest.mkdir(exist_ok=True,parents=True)
    net=NoduleUNet()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['model']);del state
    net.eval()
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    path=dest/'probabilities.npy'
    probs=np.lib.format.open_memmap(path.with_suffix('.tmp'),mode='w+',dtype='float32',shape=(len(pool.val),2,pool.n,pool.n))
    with torch.inference_mode():
        for j in range(0,len(pool.val),4):
            x,_,_=pool.batch(pool.val[j:j+4],'gray','cpu')
            x=(x/255-mean)/std
            g=net(x).sigmoid();f=net(x.flip([-1])).sigmoid().flip([-1])
            probs[j:j+len(x),0]=((g+f)/2)[:,0].numpy()
            probs[j:j+len(x),1]=(g-f).abs()[:,0].numpy()
    probs.flush();del probs
    path.with_suffix('.tmp').replace(path)
    write(dest/'probabilities.json',dict(ids=pool.val.tolist(),checkpoint_sha256=digest(checkpoint),
          manifest_sha256=digest(data/'manifest.json'),array_sha256=digest(path)))
    print('Verified source and cached probability features for',len(pool.val),'images.')

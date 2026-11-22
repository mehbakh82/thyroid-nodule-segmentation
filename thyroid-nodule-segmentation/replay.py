"""Verify both saved checkpoints against every stored validation mask."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from model import NoduleUNet
from train import ARMS,BATCH,Pool,digest


def replay(data,run):
    torch.set_num_threads(2)
    spec=json.loads((run/'run.json').read_text()); device=spec['device']
    pool=Pool(data)
    assert spec['manifest']==digest(data/'manifest.json')
    means=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    stds=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    counts={}
    for arm in ARMS:
        out=run/arm
        receipt=json.loads((out/'evaluation.json').read_text())
        assert digest(out/'checkpoint.pt')==receipt['checkpoint_sha256']
        assert digest(out/'predictions.npz')==receipt['predictions_sha256']
        ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=NoduleUNet();model.load_state_dict(ck['model']);del ck
        model=model.to(device).eval()
        with np.load(out/'predictions.npz') as f:
            ids=f['ids'];expected=f['bits']
        assert np.array_equal(ids,pool.val)
        with torch.inference_mode():
            for begin in range(0,len(ids),BATCH):
                ix=ids[begin:begin+BATCH]
                x,_,v=pool.batch(ix,arm,device)
                p=(model((x/255-means)/stds)>=0)&(v>0)
                np.testing.assert_array_equal(np.packbits(p[:,0].cpu().numpy(),axis=2),expected[begin:begin+BATCH])
        counts[arm]=len(ids)
        del model
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    result=replay(a.data,a.run)
    (a.run/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

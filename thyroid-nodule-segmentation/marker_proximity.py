"""Count false foreground near visually identified marks on a small image-only review sample."""
import json
from pathlib import Path
import numpy as np


def check(data,run,order_file,location_file):
    order=json.loads(order_file.read_text())
    points=json.loads(location_file.read_text())['points']
    truth=np.load(data/'masks.npy',mmap_mode='r')
    masks={}
    for arm in ['gray','edge']:
        with np.load(run/arm/'predictions.npz') as f:
            masks[arm]={int(i):np.unpackbits(b,axis=-1)[:,:truth.shape[-1]].astype(bool)
                        for i,b in zip(f['ids'],f['bits'])}
    yy,xx=np.indices(truth.shape[1:])
    cases=[]
    for slot,coords in points.items():
        i=order[int(slot)-1]
        area=np.logical_or.reduce([(xx-x)**2+(yy-y)**2<=16 for x,y in coords])
        eligible=area & ~truth[i].astype(bool)
        cases.append({arm:int(np.sum(eligible & masks[arm][i])) for arm in masks}
                     | dict(background=int(eligible.sum()),n_points=len(coords)))
    counts={k:sum(c[k] for c in cases) for k in cases[0]}
    return dict(images=len(cases),radius_px=4,**counts,
                more_false_foreground=sum(c['edge']>c['gray'] for c in cases),
                less_false_foreground=sum(c['edge']<c['gray'] for c in cases),
                equal=sum(c['edge']==c['gray'] for c in cases))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--order',type=Path,required=True)
    args=p.parse_args()
    result=check(args.data,args.run,args.order,Path(__file__).with_name('marker_locations.json'))
    (args.run/'marker-proximity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

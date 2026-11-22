"""Conservative near-copy screen. It cannot link clinical encounters or recover patient IDs."""
import argparse,json
from pathlib import Path
from itertools import combinations
import numpy as np
from PIL import Image

def signatures(files):
    hashes=[];arrays=[]
    for p in files:
        with Image.open(p) as f:
            im=f.convert('L')
            a=np.asarray(im.resize((9,8),Image.Resampling.BILINEAR))
            hashes.append(int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big'))
            thumb=np.asarray(im.resize((64,64),Image.Resampling.BILINEAR),dtype='f4').flatten()
            thumb-=thumb.mean();arrays.append(thumb/max(np.linalg.norm(thumb),1.e-6))
    return hashes,arrays

def compare(left,right,within=False):
    hl,al=left;hr,ar=right;rows=matches=0;ls=set();rs=set()
    for i,lh in enumerate(hl):
        for j,rh in enumerate(hr):
            if within and j<=i:continue
            if (lh^rh).bit_count()<=4:
                rows+=1
                if al[i]@ar[j]>=.995:
                    matches+=1;ls.add(i);rs.add(j)
    return dict(candidate_pairs=rows,high_similarity_pairs=matches,left_flagged=len(ls),right_flagged=len(rs))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('tn3k',type=Path);p.add_argument('ddti',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    ds=[sorted((a.tn3k/d).glob('*.jpg')) for d in ['trainval-image','test-image']]
    ds.append([a.ddti/'img'/f.name for f in sorted((a.ddti/'label').glob('*.png'))])
    train,test,dd=map(signatures,ds)
    folds=[json.loads(p.read_text()) for p in sorted(a.tn3k.glob('*fold*.json'))]
    splits=[dict(train=len(f['train']),val=len(f['val']),indices_disjoint=set(f['train']).isdisjoint(f['val'])) for f in folds]
    result=dict(trainval_vs_official_test=compare(train,test),trainval_within=compare(train,train,True),
                TN3K_trainval_vs_DDTI=compare(train,dd),TN3K_test_vs_DDTI=compare(test,dd),
                folds=splits,validation_indices_cover_all_trainval_once=sorted(i for f in folds for i in f['val'])==list(range(len(ds[0]))),
                criterion='64-bit difference hash with Hamming distance <=4 then correlation of 64x64 grayscale images >=0.995. Flag for review; no registration or crop/flip search.')
    a.out.write_text(json.dumps(result,indent=2)+'\n')

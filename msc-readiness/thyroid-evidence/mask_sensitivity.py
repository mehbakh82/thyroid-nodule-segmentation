"""Development-label sensitivity to grayscale decoding on TN3K, stratified by lesion size."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

def dice(a,b):return 2*np.count_nonzero(a&b)/(a.sum()+b.sum())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tn3k',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
    rows=[]
    for f in (args.tn3k/'trainval-mask').glob('*.jpg'):
        m=np.asarray(Image.open(f).convert('L'));base=m>127
        if base.any():rows.append([base.mean(),dice(base,m==m.max()),dice(base,m>63),dice(base,m>191)])
    a=np.array(rows);q1,q3=np.quantile(a[:,0],[.25,.75])
    def stats(x):
        return dict(n=len(x),median_self_dice={'max_only':float(np.median(x[:,1])),'above_63':float(np.median(x[:,2])),'above_191':float(np.median(x[:,3]))})
    args.out.write_text(json.dumps(dict(quantile_boundaries=[q1,q3],all=stats(a),smallest_quartile=stats(a[a[:,0]<=q1]),largest_quartile=stats(a[a[:,0]>=q3]),reference='grayscale >127',split='trainval'),indent=2)+'\n')

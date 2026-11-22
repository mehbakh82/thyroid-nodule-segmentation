"""Exploratory image-only screen using the fixed gray and contrast-only predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import sha


def descriptors(im):
    """Use the unpadded scan before inference; no contour, mask or filenames as model inputs."""
    im=np.asarray(im,dtype=float)/255
    low,high=np.quantile(im,[.1,.9])
    residual=im-ndi.gaussian_filter(im,1)
    grad=ndi.gaussian_gradient_magnitude(im,1)
    return [float(np.median(im)),float(high-low),float((im<=.02).mean()),
            float(np.median(abs(residual))),float(np.quantile(grad,.9))]


def run(root):
    data=root/'local/data'
    ablation=root/'runs/ablation-fold0'
    result=json.loads((root/'results/channel-ablation.json').read_bytes())
    assert sha(ablation/'paired-four.json')==result['paired_sha256']
    rows=json.loads((ablation/'paired-four.json').read_bytes())
    assert sha(data/'manifest.json')==json.loads((ablation/'run.json').read_bytes())['manifest']
    manifest=json.loads((data/'manifest.json').read_bytes())
    assert sha(data/'images.npy')==manifest['arrays']['images.npy']
    im=np.load(data/'images.npy',mmap_mode='r')
    features=[];gray=[];contrast=[]
    for r in rows:
        meta=manifest['rows'][r['i']]
        assert meta['role']=='validation' and r['file']==meta['file']
        x,y,w,h=meta['box']
        features.append(descriptors(im[r['i'],y:y+h,x:x+w]))
        gray.append(r['gray']['dice']);contrast.append(r['contrast']['dice'])
    X=np.asarray(features)
    delta=np.array(contrast)-gray
    order=sorted(range(len(rows)),key=lambda i:hashlib.sha256(
        ('premeeting-contrast-selection-1|'+rows[i]['file']).encode()).digest())
    fold=np.empty(len(order),dtype=int);fold[order]=np.arange(len(order))%5
    choose=np.zeros(len(rows),dtype=bool);fixed=np.zeros_like(choose)
    for k in range(5):
        test=fold==k;train=~test
        model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
        model.fit(X[train],delta[train])
        choose[test]=model.predict(X[test])>0
        fixed[test]=delta[train].mean()>0
    selected=np.asarray(gray)+np.where(choose,delta,0.)
    best=np.maximum(gray,contrast)
    rng=np.random.default_rng(241123)
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    def compare(d):
        return dict(mean_points=100*float(np.mean(d)),
                    conditional_interval_points=(100*np.quantile(d[draws].mean(axis=1),[.025,.975])).tolist())
    return dict(images=len(rows),features=['median intensity','90th minus 10th percentile intensity',
                'fraction at or below .02','median absolute Gaussian residual','90th percentile gradient magnitude'],
                folds=np.bincount(fold).tolist(),ridge_alpha=100,apply_contrast=int(choose.sum()),
                selected_higher=int(np.count_nonzero(choose&(delta>0))),
                gray=float(np.mean(gray)),contrast=float(np.mean(contrast)),selected=float(selected.mean()),
                oracle=float(best.mean()),oracle_gain_vs_gray_points=100*float(np.mean(best-gray)),
                selected_vs_gray=compare(selected-gray),selected_vs_contrast=compare(selected-contrast),
                selected_vs_other_fold_mean_choice=compare(selected-(np.array(gray)+np.where(fixed,delta,0.))),
                at_least_five_points_better=int((delta>=.05).sum()),at_least_five_points_worse=int((delta<=-.05).sum()),
                by_fold_gain_vs_gray_points=[100*float(np.mean((selected-gray)[fold==k])) for k in range(5)],
                input_sha256=result['paired_sha256'])


if __name__=='__main__':
    p=Path(__file__).resolve().parent
    result=run(p)
    (p/'results/contrast-selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

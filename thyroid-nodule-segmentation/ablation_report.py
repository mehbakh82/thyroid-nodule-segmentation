"""Four-arm paired factorial analysis on the original 576 TN3K validation images."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate import measure,sha

ARMS=('gray','contrast','gradient','edge')


def contrasts(dice):
    gray,contrast,gradient,combined=dice.T
    return {
        'contrast_minus_gray':contrast-gray,
        'gradient_minus_gray':gradient-gray,
        'combined_minus_gray':combined-gray,
        'combined_minus_gradient':combined-gradient,
        'combined_minus_contrast':combined-contrast,
        'contrast_minus_gradient':contrast-gradient,
        'interaction':combined-contrast-gradient+gray,
        'contrast_main_effect':((contrast-gray)+(combined-gradient))/2,
        'gradient_main_effect':((gradient-gray)+(combined-contrast))/2}


def summarize(rows):
    if not rows:return {'images':0}
    dice=np.array([[r[a]['dice'] for a in ARMS] for r in rows])
    rng=np.random.default_rng(241122)
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    effects={}
    for name,delta in contrasts(dice).items():
        bounds=np.quantile(delta[draws].mean(axis=1),[.025,.975])
        effects[name]=dict(mean=float(delta.mean()),interval=bounds.tolist(),
                           median=float(np.median(delta)),higher=int((delta>0).sum()),
                           lower=int((delta<0).sum()),tied=int((delta==0).sum()))
    means={}
    for a in ARMS:
        means[a]={k:float(np.mean([r[a][k] for r in rows])) for k in ('dice','surface_dice','hd95','iou')}
        means[a].update(zero_overlap=sum(r[a]['miss'] for r in rows),
                        empty=sum(r[a]['empty'] for r in rows),
                        rescues_vs_gray=sum(r['gray']['miss'] and r[a]['dice']>=.5 for r in rows),
                        harms_vs_gray=sum(r[a]['miss'] and r['gray']['dice']>=.5 for r in rows))
    return dict(images=len(rows),means=means,effects=effects)


def report(data,prior,new,marker_file):
    read=lambda p:json.loads(p.read_bytes())
    base=read(prior/'run.json')
    spec=read(new/'run.json')
    assert all(spec[k]==v for k,v in base.items())
    assert spec['baseline_run']==sha(prior/'run.json') and base['manifest']==sha(data/'manifest.json')
    ref=read(prior/'summary.json')
    assert ref['paired_sha256']==sha(prior/'paired.json') and ref['marker_review_sha256']==sha(marker_file)
    marker=read(marker_file)['labels']
    orig={r['i']:r for r in read(prior/'paired.json')}
    meta=read(data/'manifest.json')
    assert meta['arrays']['masks.npy']==sha(data/'masks.npy')
    val=[r for r in meta['rows'] if r['role']=='validation'];ids=[r['i'] for r in val]
    assert len(val)==len(set(ids))==base['val']==576
    truth=np.load(data/'masks.npy',mmap_mode='r')
    outputs={};receipts={}
    for arm in ARMS:
        run=prior if arm in ('edge','gray') else new
        receipt=read(run/arm/'evaluation.json');file=run/arm/'predictions.npz'
        assert receipt['predictions_sha256']==sha(file)
        assert receipt['arm']==arm and receipt['epochs']==base['epochs'] and read(run/'replay.json')[arm]==len(val)
        with np.load(file) as saved:
            np.testing.assert_array_equal(saved['ids'],ids)
            outputs[arm]=np.unpackbits(saved['bits'],axis=-1)[:,:,:meta['size']].astype(bool)
        receipts[arm]=receipt
    assert len({r['initial_sha256'] for r in receipts.values()})==1
    assert len({r['parameters'] for r in receipts.values()})==1
    rows=[]
    for j,r in enumerate(val):
        i=r['i'];x,y,w,h=r['box'];s=np.s_[y:y+h,x:x+w]
        case=dict(i=i,file=r['file'],small=r['area']<=meta['subgroup']['area_q25'],
                  weak=r['margin'] is not None and r['margin']<=meta['subgroup']['margin_q25'],
                  marker=marker.get(str(i)))
        for arm in ARMS:
            case[arm]=measure(outputs[arm][j][s],truth[i][s].astype(bool))
            if arm in ('gray','edge'):assert case[arm]==orig[i][arm]
        rows.append(case)
    paired=new/'paired-four.json'
    paired.write_text(json.dumps(rows,allow_nan=False)+'\n')
    thermal=[json.loads(l) for l in (new/'thermal.jsonl').read_text().splitlines()]
    telemetry=dict(observations=len(thermal),states=dict(Counter(r['thermal'] for r in thermal)),
                   ac_observations=sum(r['ac_power'] for r in thermal),
                   peak_sampled_driver_bytes=max(r['driver_bytes'] for r in thermal),
                   elapsed_seconds=thermal[-1]['time']-thermal[0]['time'])
    result=dict(overall=summarize(rows),subgroups={},receipts=receipts,paired_sha256=sha(paired),
                run_sha256=sha(new/'run.json'),telemetry=telemetry,training_seconds={})
    for a in ARMS:
        run=prior if a in ('gray','edge') else new
        hist=read(run/a/'history.json')
        assert [h['epoch'] for h in hist]==list(range(1,base['epochs']+1))
        result['training_seconds'][a]=sum(h['seconds'] for h in hist)
    for label,test in [('small',lambda r:r['small']),('weak',lambda r:r['weak']),
                       ('marked_review',lambda r:r['marker']=='clear'),
                       ('no_obvious_marks_review',lambda r:r['marker']=='no_obvious'),
                       ('uncertain_review',lambda r:r['marker']=='uncertain')]:
        result['subgroups'][label]=summarize([r for r in rows if test(r)])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('local/data'))
    p.add_argument('--prior',type=Path,default=Path('runs/source-fold0'))
    p.add_argument('--new',type=Path,default=Path('runs/ablation-fold0'))
    p.add_argument('--markers',type=Path,default=Path('local/blind/marker-review.json'))
    p.add_argument('--out',type=Path,default=Path('results/channel-ablation.json'))
    a=p.parse_args()
    result=report(a.data,a.prior,a.new,a.markers)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['overall'],indent=2))

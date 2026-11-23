"""Verify model ranking scores from saved masks and replay input channels on source images."""
import csv
from math import fsum
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

from model_ranking import ARMS, METRICS, PROTOCOL, RUN, network, predict_grid
from pairs import ARCHIVE, HERE, OUT, ROOT, file_hash, load, read_png
from train import Pool


def verify_source():
    pool = Pool(ROOT/"local/data")
    # The original loader builds per-arm channels before padding.
    indices = pool.val[np.linspace(0,len(pool.val)-1,6,dtype=int)]
    with torch.inference_mode():
        for arm in ARMS:
            receipt = load(RUN/arm/"evaluation.json")
            net = network(arm,receipt["run_key"],receipt)
            with np.load(RUN/arm/"predictions.npz") as originals:
                for index in indices:
                    x,y,w,h = pool.rows[index]["box"]
                    rgb,_,_ = pool.batch([index],arm,"cpu")
                    mu = torch.tensor([.485,.456,.406])[None,:,None,None]
                    sigma = torch.tensor([.229,.224,.225])[None,:,None,None]
                    normalized = (rgb/255-mu)/sigma
                    logits = net(normalized)
                    native = Image.fromarray(np.array(pool.im[index,y:y+h,x:x+w]))
                    source = (logits[0,0,y:y+h,x:x+w]>=0).numpy()
                    pos = np.flatnonzero(originals["ids"]==index).item()
                    stored = np.unpackbits(originals["bits"][pos],axis=1)[y:y+h,x:x+w]
                    np.testing.assert_array_equal(source,stored)
                    two_view = (logits.sigmoid()+net(normalized.flip([-1])).sigmoid().flip([-1]))/2
                    np.testing.assert_array_equal(predict_grid(net,arm,native),
                        (two_view[0,0,y:y+h,x:x+w]>=.5).numpy())
    print("Six source images per model reproduced archived single-view masks and original-loader two-view controls.")


def verify_scores():
    with (OUT/"model-ranking.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = load(HERE/"model-ranking-results.json")
    for key,path in (("scores_sha256",OUT/"model-ranking.csv"),
                     ("predictions_sha256",OUT/"model-ranking-predictions.npz"),("protocol_sha256",PROTOCOL),
                     ("code_sha256",HERE/"model_ranking.py"),("archive_sha256",ARCHIVE)):
        assert file_hash(path)==result[key]
    assert len({r["pair"] for r in rows})==len(rows)==76
    y,x = np.ogrid[-2:3,-2:3]
    disk = x*x+y*y<=4
    with np.load(OUT/"model-ranking-predictions.npz") as preds, zipfile.ZipFile(ARCHIVE) as archive:
        for row in rows:
            size = int(row["width"]),int(row["height"])
            for ref in "AB":
                target = np.asarray(read_png(archive,"mask",row["id_"+ref]).convert("L").resize(size,Image.Resampling.NEAREST))>0
                target_edge = target ^ ndi.binary_erosion(target)
                for arm in ARMS:
                    pred = np.unpackbits(preds[row["pair"]+"_"+arm],axis=1)[:,:size[0]].astype(bool)
                    edge = pred ^ ndi.binary_erosion(pred)
                    matches = (edge & ndi.binary_dilation(target_edge,structure=disk)).sum()
                    matches += (target_edge & ndi.binary_dilation(edge,structure=disk)).sum()
                    surface = matches/(edge.sum()+target_edge.sum())
                    region = 2*np.count_nonzero(pred & target)/(pred.sum()+target.sum())
                    np.testing.assert_allclose([surface,region],
                        [float(row[f"{m}_{arm}_{ref}"]) for m in METRICS],atol=1e-12,rtol=0)
                for m in METRICS:
                    delta = 100*(float(row[f"{m}_edge_{ref}"])-float(row[f"{m}_gray_{ref}"]))
                    assert abs(delta-float(row[f"{m}_delta_{ref}"]))<1e-10
    for subset, value in (("all",None),("flagged","False"),("same_target","True")):
        selected = [r for r in rows if value is None or r["same_target"]==value]
        for m in METRICS:
            s = result["groups"][subset][m]
            deltas = [[float(r[f"{m}_delta_{ref}"]) for ref in "AB"] for r in selected]
            for i,(key,extreme) in enumerate((("lower",min),("upper",max))):
                chosen = [float(r[f"{m}_delta_"+r[f"{m}_{key}_choice"]]) for r in selected]
                np.testing.assert_allclose(chosen,[extreme(d) for d in deltas])
                assert abs(fsum(chosen)/len(chosen)-s["range_points"][i])<1e-10
            assert sum(min(d)<-1e-9 and max(d)>1e-9 for d in deltas)==s["opposite"]
            assert sum(min(d)>1e-9 for d in deltas)==s["edge_wins_both"]
            assert sum(max(d)<-1e-9 for d in deltas)==s["gray_wins_both"]
            assert sum(min(map(abs,d))<1e-9 for d in deltas)==s["tie_involved"]
            for cutoff in (1,2,5):
                assert sum(min(d)<=-cutoff and max(d)>=cutoff for d in deltas)==s["material_at"][str(cutoff)]
            assert abs(fsum(d[0] for d in deltas)/len(selected)-s["all_A"])<1e-10
            assert abs(fsum(d[1] for d in deltas)/len(selected)-s["all_B"])<1e-10
    print("Independently verified 304 model/reference scores for both metrics, per-case counts and attained cohort bounds.")


if __name__=="__main__":
    torch.set_num_threads(2)
    verify_scores()
    verify_source()

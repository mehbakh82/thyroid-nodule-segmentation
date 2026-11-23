"""Frozen two-model comparison across alternative reference masks on identical scans."""
from collections import Counter
import csv
import hashlib
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from pairs import (ARCHIVE, ARCHIVE_SHA, HERE, OUT, ROOT, NoduleUNet, file_hash, group_masks,
                   identity, load, pairing, save, score)
from train import channels

ARMS = ("gray", "edge")
METRICS = ("surface2", "region")
RUN = ROOT / "runs/source-fold0"
PROTOCOL = HERE / "MODEL-RANKING-PROTOCOL.md"


def network(arm, run_key, receipt):
    checkpoint = RUN / arm / "checkpoint.pt"
    if file_hash(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed for {arm}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["arm"] != arm or state["run_key"] != run_key or state["initial"] != receipt["initial_sha256"]:
        raise ValueError(f"Training record mismatch for {arm}")
    model = NoduleUNet()
    model.load_state_dict(state["model"])
    return model.eval()


def predict_grid(model, arm, image):
    width, height = (max(1,round(d*224/max(image.size))) for d in image.size)
    x,y = (224-width)//2, (224-height)//2
    gray = np.asarray(image.resize((width,height),Image.Resampling.BILINEAR))
    tensor = torch.zeros((1,3,224,224))
    tensor[0,:,y:y+height,x:x+width] = torch.from_numpy(channels(gray,arm))
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    std = torch.tensor([.229,.224,.225])[None,:,None,None]
    tensor = (tensor/255-mean)/std
    original = model(tensor).sigmoid()
    mirror = model(tensor.flip([-1])).sigmoid().flip([-1])
    average = (original+mirror)/2
    return average[0,0,y:y+height,x:x+width].numpy() >= .5


def summarize(rows, metric):
    prefix = metric + "_delta_"
    gains = np.array([[r[prefix+"A"],r[prefix+"B"]] for r in rows])
    outcomes = Counter(pairing(pair) for pair in gains)
    lo = gains.min(axis=1)
    hi = gains.max(axis=1)
    arm_means = {arm:{ref:float(np.mean([r[f"{metric}_{arm}_{ref}"] for r in rows])) for ref in "AB"}
                 for arm in ARMS}
    return dict(n=len(rows),mean_scores=arm_means,edge_wins_both=outcomes["both improve"],
                gray_wins_both=outcomes["both worsen"],opposite=outcomes["opposite"],
                tie_involved=outcomes["tie involved"],
                material_at={str(k):int(((lo<=-k)&(hi>=k)).sum()) for k in (1,2,5)},
                all_A=float(gains[:,0].mean()),all_B=float(gains[:,1].mean()),
                equal_weight=float(gains.mean()),range_points=[float(lo.mean()),float(hi.mean())])


def main():
    torch.set_num_threads(2)
    prior = load(HERE/"results.json")
    previous_rows = OUT/"per-pair.json"
    training = load(RUN/"run.json")
    run_key = load(RUN/"gray/evaluation.json")["run_key"]
    receipts = {arm:load(RUN/arm/"evaluation.json") for arm in ARMS}
    if (file_hash(ARCHIVE)!=ARCHIVE_SHA or file_hash(previous_rows)!=prior["per_pair_sha256"]
        or file_hash(HERE/"visual-review.json")!=prior["visual_review_sha256"]
        or file_hash(ROOT/"train.py")!=training["trainer"] or file_hash(ROOT/"model.py")!=training["model"]
        or cv2.__version__!=training["opencv"] or receipts["gray"]["checkpoint_sha256"]!=prior["checkpoint_sha256"]):
        raise ValueError("Frozen sources, preprocessing, or case records changed")
    if len({r["initial_sha256"] for r in receipts.values()})!=1:
        raise ValueError("Models have different initial parameters")
    nets = {arm:network(arm,run_key,receipts[arm]) for arm in ARMS}
    previous = load(previous_rows)
    if len(previous)!=76 or sum(r["same_target"] for r in previous)!=69 or any(r["unscorable"] for r in previous):
        raise ValueError("Unexpected reference review or exclusions")
    rows, packed = [], {}
    with torch.inference_mode(), zipfile.ZipFile(ARCHIVE) as z:
        for old in previous:
            image, native = group_masks(z,old["ids"])
            if identity(image)!=old["image_hash"]:
                raise ValueError("Changed scan " + old["pair"])
            preds = {arm:predict_grid(nets[arm],arm,image) for arm in ARMS}
            gray = preds["gray"]
            if list(gray.shape[::-1])!=old["grid"] or hashlib.sha256(gray.tobytes()).hexdigest()!=old["baseline_sha256"]:
                raise ValueError("Grayscale replay mismatch on " + old["pair"])
            row = dict(pair=old["pair"],id_A=old["ids"][0],id_B=old["ids"][1],same_target=old["same_target"],
                       width=gray.shape[1],height=gray.shape[0])
            for arm in ARMS:
                packed[row["pair"]+"_"+arm] = np.packbits(preds[arm],axis=1)
            for j,(ref,mask) in enumerate(zip("AB",native,strict=True)):
                target = np.asarray(Image.fromarray(mask).resize(gray.shape[::-1],Image.Resampling.NEAREST))
                for arm,pred in preds.items():
                    region,surface = score(pred,target)
                    row[f"region_{arm}_{ref}"] = region
                    row[f"surface2_{arm}_{ref}"] = float(surface[1])
                for metric in METRICS:
                    if abs(row[f"{metric}_gray_{ref}"] - old["scores"][j][metric])>1e-12:
                        raise ValueError("Grayscale score mismatch")
                    row[f"{metric}_delta_{ref}"] = 100*(row[f"{metric}_edge_{ref}"]-row[f"{metric}_gray_{ref}"])
            for metric in METRICS:
                a,b = (row[f"{metric}_delta_{ref}"] for ref in "AB")
                row[f"{metric}_lower_choice"] = "A" if a<=b else "B"
                row[f"{metric}_upper_choice"] = "A" if a>=b else "B"
            rows.append(row)
    score_path = OUT/"model-ranking.csv"
    with score_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    masks_path = OUT/"model-ranking-predictions.npz"
    np.savez_compressed(masks_path,**packed)
    groups = {"same_target":[r for r in rows if r["same_target"]],
              "flagged":[r for r in rows if not r["same_target"]],"all":rows}
    summary = dict(groups={g:{m:summarize(group,m) for m in METRICS} for g,group in groups.items()},
                   archive_sha256=ARCHIVE_SHA,prior_rows_sha256=file_hash(previous_rows),
                   checkpoint_sha256={a:r["checkpoint_sha256"] for a,r in receipts.items()},
                   trainer_sha256=training["trainer"],model_sha256=training["model"],
                   protocol_sha256=file_hash(PROTOCOL),code_sha256=file_hash(Path(__file__)),
                   scores_sha256=file_hash(score_path),predictions_sha256=file_hash(masks_path))
    save(HERE/"model-ranking-results.json",summary)
    print(summary["groups"]["same_target"])


if __name__ == "__main__":
    main()

"""Recompute key follow-up results from the public tables, without imaging dependencies."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
TABLES = HERE / "case-results"


def read(path):
    return json.loads(path.read_bytes())


def equal(value, expected):
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Expected {expected}, computed {value}")


def verify():
    margin = read(HERE/"results.json")
    path = TABLES/"tn3k-margin.json"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=margin["per_image_sha256"]:
        raise ValueError("Margin scores differ from original receipt")
    rows = read(path)
    localized = [r for r in rows if r["baseline"]["above127"]["dice"]>=.5]
    equal(len(localized),margin["localized"])
    for method in margin["methods"]:
        for decoding in ("above127", "only255"):
            scores = margin["methods"][method][decoding]
            equal(fmean(r[method][decoding]["s2"] for r in localized),scores["localized_mean_surface2"])

    codec = read(TABLES/"tnui-codec.json")
    for name,stats in read(HERE/"tnui-control/results.json")["overall"].items():
        equal(fmean(r[name]["shrink_delta_points"] for r in codec),stats["erosion_change_points"]["mean"])
    discovery=read(TABLES/"tnui-localization.json")
    base=read(HERE/"tnui-location/results.json")
    for arm,stats in base["methods"].items():
        equal(fmean(r["scores"][arm]["dice"] for r in discovery),stats["dice"])
    equal(len(read(TABLES/"tnui-border-discovery.json")),len(discovery))
    for arm,stats in read(HERE/"tnui-location/border-repeat.json")["methods"].items():
        equal(fmean(r[arm]["dice"] for r in read(TABLES/"tnui-border-repeat.json")),stats["mean_dice"])

    field=read(TABLES/"tnscui-field.json")
    transfer=read(HERE/"tnscui-transfer/results.json")
    equal(len({r["record"] for r in field}),transfer["inventory"])
    equal(len({r["scan"] for r in field}),transfer["unique_images"])
    equal(sum(20*r["outside"]>r["target"] for r in field),transfer["all_safety"]["above_five_percent"])
    for arm,stats in transfer["performance"]["arms"].items():
        equal(fmean(r[arm]["dice"] for r in read(TABLES/"tnscui-model.json")),stats["mean_dice"])
    boundary=read(TABLES/"tnscui-boundary.json")
    equal(sum(r["adaptive"]["dice"]<r["whole"]["dice"] for r in boundary),
          read(HERE/"tnscui-transfer/boundary-results.json")["dice_worsened"])

    same=[r for r in read(TABLES/"paired-edit-scores.json") if r["same_target"]]
    expected=read(HERE/"paired-references/results.json")["subsets"]["apparent_same_target"]["surface_delta"]
    equal(sum(min(r["surface_delta"])<=-2 and max(r["surface_delta"])>=2 for r in same),expected["reversal_min_points"]["2"])
    ranking=read(HERE/"paired-references/model-ranking-results.json")
    with (TABLES/"model-ranking.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    for group,subset in (("all",rows),("same_target",[r for r in rows if r["same_target"]=="True"]),
                         ("flagged",[r for r in rows if r["same_target"]=="False"])):
        for metric,stats in ranking["groups"][group].items():
            gains=[tuple(100*(float(r[f"{metric}_edge_{k}"])-float(r[f"{metric}_gray_{k}"])) for k in "AB") for r in subset]
            equal(len(gains),stats["n"])
            equal(fmean(map(min,gains)),stats["range_points"][0])
            equal(fmean(map(max,gains)),stats["range_points"][1])
            equal(sum(min(d)<=-2 and max(d)>=2 for d in gains),stats["material_at"]["2"])
    packed=TABLES/"model-ranking-predictions.npz"
    if hashlib.sha256(packed.read_bytes()).hexdigest()!=ranking["predictions_sha256"]:
        raise ValueError("Changed model predictions")
    with ZipFile(packed) as z:
        if z.testzip() or set(z.namelist())!={r["pair"]+"_"+a+".npy" for r in rows for a in ("gray","edge")}:
            raise ValueError("Incomplete prediction archive")
    print("Verified margin, encoding, localization, field-transfer and paired-reference results, including both ranking bounds.")


if __name__=="__main__":
    verify()

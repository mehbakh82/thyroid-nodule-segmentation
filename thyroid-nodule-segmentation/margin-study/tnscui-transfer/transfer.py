"""Replay the frozen TNUI display filter on TN-SCUI2020 training images."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "tnui-location"
sys.path.insert(0, str(PREVIOUS))
from border_check import scan_bounds
from pilot import ROOT, NoduleUNet, file_hash, identity, native_prediction

LOCAL = ROOT / "local/tnscui-transfer"
ARCHIVE_SHA = "da8996ea93cd49b1269a3906098e61c859800d9f2f9ff16f1f73b36d8eb2e284"
RULE_SHA = "6b7b0ac9f3f42367ccbd5565b0b307372ee90c59875db1829cb4f90923dff4ed"
SALT = "tnscui-field-v1/"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_png(archive, folder, name):
    entry = f"TNSC2020/{folder}/{name}"
    with Image.open(io.BytesIO(archive.read(entry))) as im:
        if im.format != "PNG":
            raise ValueError(f"Unexpected encoding in {entry}")
        return im.copy()


def inventory(archive):
    csv_rows = list(csv.DictReader(archive.read("TNSC2020/train.csv").decode().splitlines()))
    names = [r["ID"] for r in csv_rows]
    expected = {"TNSC2020/train.csv"}
    expected.update(f"TNSC2020/{folder}/{n}" for folder in ("image", "mask") for n in names)
    files = [i.filename for i in archive.infolist() if not i.is_dir()]
    if len(names) != 3644 or len(set(names)) != len(names) or sorted(files) != sorted(expected):
        raise ValueError("Training CSV and image/mask inventory disagree")
    return sorted(names)


def previous_images(tn3k):
    source, tnui = set(), set()
    source_rows = json.loads((ROOT / "local/data/manifest.json").read_bytes())["rows"]
    if len(source_rows) != 2879:
        raise ValueError("Unexpected TN3K development partition")
    for r in source_rows:
        p = tn3k / "trainval-image" / r["file"]
        if file_hash(p) != r["image_sha256"]:
            raise ValueError(f"TN3K image changed: {p.name}")
        with Image.open(p) as im:
            source.add(identity(im))
    count = 0
    for sub in ("tnui-control", "tnui-location", "tnui-border-repeat"):
        base = ROOT / "local" / sub
        manifest = json.loads((base / "sample.json").read_bytes())
        for r in manifest["rows"]:
            with Image.open(base / "images" / r["name"]) as im:
                tnui.add(identity(im))
            count += 1
    if count != 192:
        raise ValueError("Unexpected TNUI development partition")
    return source, tnui


def inspect_case(archive, name):
    image, label = (read_png(archive, folder, name) for folder in ("image", "mask"))
    if image.size != label.size:
        return None, "shape mismatch"
    if label.mode not in ("1", "L"):
        return None, "unsupported mask mode " + label.mode
    histogram = label.convert("L").histogram()
    positive = [i for i in range(1, 256) if histogram[i]]
    if len(positive) != 1 or not histogram[0]:
        return None, "nonbinary or empty mask"
    gray = image.convert("L")
    mask = np.asarray(label.convert("L")) > 0
    profile = mask.sum(axis=1)
    low, high = np.flatnonzero(profile)[[0, -1]]
    top, bottom = scan_bounds(np.asarray(gray))
    gap = int(min(low - top, bottom - 1 - high))
    return dict(id=name, width=image.width, height=image.height, mode=image.mode,
                mask_foreground_value=positive[0], image_hash=identity(gray),
                mask_hash=hashlib.sha256(mask.tobytes()).hexdigest(),
                target=int(profile.sum()), outside=int(profile[:top].sum() + profile[bottom:].sum()),
                top=top, bottom=bottom, gap=gap, boundary=50 * gap <= image.height), None


def safety(rows):
    if not rows:
        return {"n": 0}
    target = sum(r["target"] for r in rows)
    outside = sum(r["outside"] for r in rows)
    losses = [r["outside"] / r["target"] for r in rows]
    return dict(n=len(rows), trimmed=sum(r["top"] > 0 or r["bottom"] < r["height"] for r in rows),
                target=target, outside=outside, affected=sum(x > 0 for x in losses),
                above_one_percent=sum(100*r["outside"] > r["target"] for r in rows),
                above_five_percent=sum(20*r["outside"] > r["target"] for r in rows),
                retained_fraction=1-outside/target, max_loss=max(losses))


def score(pred, mask, top, bottom):
    tp = int(np.count_nonzero(pred[top:bottom] & mask[top:bottom]))
    fp = int(np.count_nonzero(pred[top:bottom])) - tp
    fn = int(mask.sum()) - tp
    return dict(tp=tp, fp=fp, fn=fn, dice=2*tp/(2*tp+fp+fn), sensitivity=tp/(tp+fn))


def inference(archive, selected, fractions, checkpoint):
    model = NoduleUNet()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    rows = []
    with torch.inference_mode():
        for r in selected:
            image = read_png(archive, "image", r["id"]).convert("L")
            mask = np.asarray(read_png(archive, "mask", r["id"]).convert("L")) > 0
            prediction = native_prediction(model, image, (0, 0, *image.size))
            arms = dict(whole=(0, image.height), adaptive=(r["top"], r["bottom"]),
                        fixed=np.rint(fractions * image.height).astype(int))
            result = {k: score(prediction, mask, *limits) for k, limits in arms.items()}
            rows.append(dict(id=r["id"], **result))
    return rows


def performance(rows):
    arms = {}
    for arm in ("whole", "adaptive", "fixed"):
        scores = [r[arm] for r in rows]
        arms[arm] = dict(mean_dice=float(np.mean([m["dice"] for m in scores])),
                         mean_sensitivity=float(np.mean([m["sensitivity"] for m in scores])),
                         misses=sum(m["tp"] == 0 for m in scores),
                         fp=sum(m["fp"] for m in scores), tp=sum(m["tp"] for m in scores))
    draws = np.random.default_rng(231130).integers(len(rows), size=(5000, len(rows)))
    differences = {}
    for a, b in (("adaptive", "whole"), ("fixed", "whole"), ("adaptive", "fixed")):
        delta = np.array([100*(r[a]["dice"]-r[b]["dice"]) for r in rows])
        differences[a+"_minus_"+b] = dict(mean_points=float(delta.mean()),
            ci_points=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
            improved=int(sum(delta > 0)), worsened=int(sum(delta < 0)), tied=int(sum(delta == 0)),
            removed_fp=arms[b]["fp"]-arms[a]["fp"], removed_tp=arms[b]["tp"]-arms[a]["tp"])
    return dict(n=len(rows), arms=arms, differences=differences)


def run(tn3k):
    torch.set_num_threads(2)
    archive_path = LOCAL / "TNSC2020.zip"
    checkpoint = ROOT / "runs/source-fold0/gray/checkpoint.pt"
    checkpoint_sha = json.loads((checkpoint.parent / "evaluation.json").read_bytes())["checkpoint_sha256"]
    if file_hash(archive_path) != ARCHIVE_SHA or file_hash(checkpoint) != checkpoint_sha:
        raise ValueError("Archive or frozen checkpoint checksum mismatch")
    if file_hash(PREVIOUS / "border_check.py") != RULE_SHA:
        raise ValueError("Frozen filter checksum mismatch")
    previous = json.loads((PREVIOUS / "border-repeat.json").read_bytes())
    fractions = np.array(previous["median_bounds_from_previous_96"])
    old_bounds = ROOT / "local/tnui-location/border-per-image.json"
    if file_hash(old_bounds) != previous["old_bounds_sha256"]:
        raise ValueError("Original fixed-band data changed")
    bounds = json.loads(old_bounds.read_bytes())
    np.testing.assert_array_equal(fractions, np.median(
        [[r["top"]/r["height"], r["bottom"]/r["height"]] for r in bounds], axis=0))
    source, tnui = previous_images(tn3k)
    rows, exclusions = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        names = inventory(archive)
        for i, name in enumerate(names, 1):
            row, reason = inspect_case(archive, name)
            if reason:
                exclusions[name] = reason
            else:
                rows.append(row)
            if i % 1000 == 0:
                print("Audited", i, flush=True)
        ordered = sorted(rows, key=lambda r: hashlib.sha256((SALT+r["id"]).encode()).digest())
        groups = defaultdict(list)
        for row in ordered:
            groups[row["image_hash"]].append(row)
        unique = [g[0] for g in groups.values()]
        eligible = [r for r in unique if r["image_hash"] not in source and r["image_hash"] not in tnui]
        selected = eligible[:160]
        if len(selected) != 160:
            raise ValueError("Insufficient eligible cases")
        save(LOCAL / "audit.json", dict(rows=rows, exclusions=exclusions))
        save(LOCAL / "sample.json", [r["id"] for r in selected])
        print("Safety audit complete. Running paired CPU inference.", flush=True)
        scores = inference(archive, selected, fractions, checkpoint)
    save(LOCAL / "per-image.json", scores)
    all_safety = safety(rows)
    sizes = Counter((r["width"], r["height"]) for r in rows)
    perf = performance(scores)
    summary = dict(archive_sha256=ARCHIVE_SHA, checkpoint_sha256=checkpoint_sha, filter_sha256=RULE_SHA,
                   protocol_sha256=file_hash(HERE / "PROTOCOL.md"), code_sha256=file_hash(Path(__file__)),
                   audit_sha256=file_hash(LOCAL / "audit.json"), sample_sha256=file_hash(LOCAL / "sample.json"),
                   per_image_sha256=file_hash(LOCAL / "per-image.json"), fixed_fractions=fractions.tolist(),
                   inventory=len(names), repeated_ids=0, exclusions=dict(Counter(exclusions.values())),
                   unique_images=len(unique), repeated_rasters=len(rows)-len(unique),
                   conflicting_mask_groups=sum(len({r["mask_hash"] for r in g}) > 1 for g in groups.values()),
                   source_overlap_unique=sum(r["image_hash"] in source for r in unique),
                   tnui_overlap_unique=sum(r["image_hash"] in tnui for r in unique),
                   eligible_for_inference=len(eligible), native_sizes=len(sizes),
                   most_common_sizes=[dict(width=w, height=h, n=n) for (w,h),n in sizes.most_common(6)],
                   modes=dict(Counter(r["mode"] for r in rows)),
                   mask_values=dict(Counter(r["mask_foreground_value"] for r in rows)),
                   all_safety=all_safety, unique_safety=safety(unique),
                   boundary_safety=safety([r for r in rows if r["boundary"]]),
                   other_safety=safety([r for r in rows if not r["boundary"]]),
                   sample_boundary=sum(r["boundary"] for r in selected), performance=perf,
                   safety_criterion_met=all_safety["retained_fraction"] >= .99 and all_safety["above_five_percent"] == 0,
                   performance_criterion_met=perf["differences"]["adaptive_minus_whole"]["ci_points"][0] > 0)
    save(HERE / "results.json", summary)
    print("Saved", HERE / "results.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tn3k", type=Path)
    run(p.parse_args().tn3k)

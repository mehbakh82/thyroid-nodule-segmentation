"""Prepare, inspect and score paired TN-SCUI2020 references at fixed model output."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tnscui-transfer"))
from transfer import ARCHIVE_SHA, LOCAL as SOURCE, ROOT, NoduleUNet, file_hash, identity, read_png, save
from scale_check import predict
from boundary_audit import score

OUT = ROOT / "local/paired-references"
ARCHIVE = SOURCE / "TNSC2020.zip"
BASE = ROOT / "runs/source-fold0/gray/checkpoint.pt"


def load(path):
    return json.loads(path.read_bytes())


def group_masks(z, names):
    images = [read_png(z, "image", name) for name in names]
    if images[0].size != images[1].size or not np.array_equal(images[0], images[1]):
        raise ValueError("Nonidentical images")
    masks = []
    for name in names:
        label = read_png(z, "mask", name)
        raw = np.asarray(label.convert("L"))
        if label.size != images[0].size or label.mode not in ("1", "L") or set(np.unique(raw)) != {0, 255}:
            raise ValueError(f"Invalid reference at {name}")
        masks.append(raw == 255)
    if np.array_equal(*masks):
        raise ValueError("References are identical")
    return images[0], masks


def flags_for(a, b, agreement):
    flags = []
    if agreement < .75:
        flags.append("low overlap")
    for label, mask, other in (("A", a, b), ("B", b, a)):
        center = tuple(np.rint(ndi.center_of_mass(mask)).astype(int))
        if not other[center]:
            flags.append(label + " centroid outside other mask")
        parts, _ = ndi.label(mask, structure=np.ones((3, 3)))
        sizes = np.bincount(parts.ravel())
        overlap = np.bincount(parts[other], minlength=len(sizes))
        if np.any((20*sizes[1:] >= mask.sum()) & (10*overlap[1:] < sizes[1:])):
            flags.append(label + " has unmatched component")
    return flags


def outline_image(image, masks, size):
    """Draw resized native contours with a readable line width."""
    thumb = image.convert("RGB")
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    edges = []
    for mask in masks:
        reduced = np.asarray(Image.fromarray(mask).resize(thumb.size, Image.Resampling.NEAREST))
        edges.append(reduced ^ ndi.binary_erosion(reduced))
    rgb = np.array(thumb)
    rgb[edges[0]] = [30, 230, 255]
    rgb[edges[1]] = [255, 146, 42]
    rgb[edges[0] & edges[1]] = [255, 255, 255]
    return Image.fromarray(rgb)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = load(HERE.parent / "tnscui-transfer/results.json")
    if file_hash(ARCHIVE) != ARCHIVE_SHA or file_hash(SOURCE/"audit.json") != prior["audit_sha256"]:
        raise ValueError("Archive or source audit changed")
    groups = defaultdict(list)
    for row in load(SOURCE/"audit.json")["rows"]:
        groups[row["image_hash"]].append(row["id"])
    paired = sorted((key, ids) for key, ids in groups.items() if len(ids)>1)
    if len(paired) != 76 or any(len(ids)!=2 for _,ids in paired):
        raise ValueError("Unexpected duplicate structure")
    rows = []
    with zipfile.ZipFile(ARCHIVE) as z:
        categories = {r["ID"]:r["CATE"] for r in csv.DictReader(z.read("TNSC2020/train.csv").decode().splitlines())}
        for index, (key, ids) in enumerate(paired):
            ids = sorted(ids, key=lambda n:int(Path(n).stem))
            image, (a,b) = group_masks(z, ids)
            if identity(image) != key:
                raise ValueError("Pixel digest mismatch")
            agreement = float(2*(a & b).sum()/(a.sum()+b.sum()))
            flags = flags_for(a,b,agreement)
            label = f"P{index+1:02}"
            rows.append(dict(pair=label, ids=ids, image_hash=key, native_dice=agreement, flags=flags,
                             class_conflict=categories[ids[0]]!=categories[ids[1]]))
            slot = index % 16
            if slot == 0:
                sheet = Image.new("RGB",(1100,820),"#ffffff")
                draw = ImageDraw.Draw(sheet)
            x = slot%4*275
            y = slot//4*205
            overlay = outline_image(image,(a,b),(265,177))
            sheet.paste(overlay,(x+(275-overlay.width)//2, y+(177-overlay.height)//2))
            draw.text((x+8,y+181),f"{label}   Dice {agreement:.2f}" + ("   FLAG" if flags else ""),
                      fill="#152e3b",font=ImageFont.load_default(size=15))
            if slot == 15 or index+1 == len(paired):
                sheet.save(OUT/f"review-{index//16+1}.png")
    save(OUT/"geometry.json",rows)
    print("Made five image-only reference review sheets; pair flags:",[(r['pair'],r['flags']) for r in rows if r['flags']])


def pairing(deltas):
    first, second = deltas
    if min(abs(first), abs(second)) < 1e-9:
        return "tie involved"
    if first*second < 0:
        return "opposite"
    return "both improve" if first>0 else "both worsen"


def quantiles(values):
    return dict(zip(("min","q25","median","q75","max"),np.quantile(values,[0,.25,.5,.75,1]).tolist()))


def summarize(rows, metric):
    if not rows:
        return {"n":0}
    deltas = [r[metric] for r in rows]
    classes = Counter(map(pairing, deltas))
    strengths = [min(map(abs,d)) for d in deltas if pairing(d)=="opposite"]
    return dict(n=len(rows), outcomes={name:classes[name] for name in ("both improve","both worsen","opposite","tie involved")},
                reversal_min_points={str(k):sum(s>=k for s in strengths) for k in (1,2,5)},
                reversal_strength_points=quantiles(strengths) if strengths else None,
                absolute_difference_of_deltas=quantiles([abs(a-b) for a,b in deltas]),
                mean_delta_across_both_references=float(np.mean(deltas)))


def evaluate():
    prior = load(HERE.parent/"tnscui-transfer/results.json")
    if file_hash(BASE)!=prior["checkpoint_sha256"] or file_hash(ARCHIVE)!=ARCHIVE_SHA:
        raise ValueError("Frozen inputs changed")
    review = load(HERE/"visual-review.json")
    geometry = load(OUT/"geometry.json")
    if file_hash(OUT/"geometry.json")!=review["geometry_sha256"] or len(geometry)!=76:
        raise ValueError("Visual review and geometry disagree")
    notes = review["visual_flags"]
    if not set(notes).issubset(r["pair"] for r in geometry):
        raise ValueError("Unknown review ID")
    torch.set_num_threads(2)
    model = NoduleUNet()
    state = torch.load(BASE,map_location="cpu",weights_only=True)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    rows = []
    with zipfile.ZipFile(ARCHIVE) as z, torch.inference_mode():
        for g in geometry:
            image, masks = group_masks(z, g["ids"])
            baseline = predict(model,image,224)[0,0].numpy()>=.5
            smaller = ndi.binary_erosion(baseline)
            grid = baseline.shape[::-1]
            refs = [np.array(Image.fromarray(m).resize(grid,Image.Resampling.NEAREST)) for m in masks]
            row = dict(**g, visual_flag=notes.get(g["pair"]), same_target=not(g["flags"] or g["pair"] in notes),
                       baseline_sha256=hashlib.sha256(baseline.tobytes()).hexdigest(), grid=list(grid))
            if not all(ref.any() for ref in refs):
                rows.append(dict(**row,unscorable=True))
                continue
            scores = []
            for ref in refs:
                d0, s0 = score(baseline,ref)
                d1, s1 = score(smaller,ref)
                scores.append(dict(region=d0,smaller_region=d1,surface2=float(s0[1]),smaller_surface2=float(s1[1])))
            rows.append(dict(**row,unscorable=False,scores=scores,localized=all(s["region"]>=.5 for s in scores),
                             surface_delta=[100*(s["smaller_surface2"]-s["surface2"]) for s in scores],
                             region_delta=[100*(s["smaller_region"]-s["region"]) for s in scores]))
    save(OUT/"per-pair.json",rows)
    valid = [r for r in rows if not r["unscorable"]]
    same = [r for r in valid if r["same_target"]]
    flagged = [r for r in valid if not r["same_target"]]
    subsets = {"all":valid,"apparent_same_target":same,"flagged":flagged,
               "localized_same_target":[r for r in same if r["localized"]]}
    summary = dict(pairs=len(rows),unscorable=len(rows)-len(valid),class_conflicts=sum(r["class_conflict"] for r in rows),
                   mask_dice=quantiles([r["native_dice"] for r in rows]),
                   subsets={k:{m:summarize(v,m) for m in ("surface_delta","region_delta")} for k,v in subsets.items()},
                   prioritize_tracing=sum(pairing(r["surface_delta"])=="opposite" and min(map(abs,r["surface_delta"]))>=2 for r in same)>=5,
                   archive_sha256=ARCHIVE_SHA,checkpoint_sha256=prior["checkpoint_sha256"],
                   protocol_sha256=file_hash(HERE/"PROTOCOL.md"),visual_review_sha256=file_hash(HERE/"visual-review.json"),
                   code_sha256=file_hash(Path(__file__)),per_pair_sha256=file_hash(OUT/"per-pair.json"))
    save(HERE/"results.json",summary)
    print(json.dumps({k:v["surface_delta"] for k,v in summary["subsets"].items()},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","evaluate"))
    if parser.parse_args().stage == "prepare":
        prepare()
    else:
        evaluate()

"""Replay both references and independently reconstruct the two-pixel surface metric."""
import hashlib
import zipfile
from collections import Counter

import numpy as np
from scipy import ndimage as ndi
import torch
from PIL import Image

from pairs import HERE, OUT, ARCHIVE, BASE, NoduleUNet, file_hash, load, predict, read_png


def check():
    summary = load(HERE/"results.json")
    for key,path in (("per_pair_sha256",OUT/"per-pair.json"),("archive_sha256",ARCHIVE),
                     ("protocol_sha256",HERE/"PROTOCOL.md"),("code_sha256",HERE/"pairs.py"),
                     ("visual_review_sha256",HERE/"visual-review.json"),("checkpoint_sha256",BASE)):
        assert file_hash(path) == summary[key], key
    rows = load(OUT/"per-pair.json")
    assert len({r["image_hash"] for r in rows}) == len(rows) == 76
    model = NoduleUNet()
    model.load_state_dict(torch.load(BASE,map_location="cpu",weights_only=True)["model"])
    model.eval()
    torch.set_num_threads(2)
    y,x = np.ogrid[-2:3,-2:3]
    disk = x*x+y*y <= 4
    counts = {s:Counter() for s in summary["subsets"]}
    observations = 0
    with zipfile.ZipFile(ARCHIVE) as z, torch.inference_mode():
        for row in rows:
            image = read_png(z,"image",row["ids"][1])
            raw = predict(model,image,224)[0,0].numpy() >= .5
            assert hashlib.sha256(raw.tobytes()).hexdigest() == row["baseline_sha256"]
            # Explicit cross-shaped shrink with zero background outside the native grid.
            padded = np.pad(raw, 1)
            shrunk = raw & padded[:-2,1:-1] & padded[2:,1:-1] & padded[1:-1,:-2] & padded[1:-1,2:]
            changes = []
            for index,name in enumerate(row["ids"]):
                target = np.asarray(read_png(z,"mask",name).convert("L").resize(raw.shape[::-1],Image.Resampling.NEAREST))>0
                assert target.any() and not row["unscorable"]
                target_edge = target ^ ndi.binary_erosion(target)
                values = []
                for pred in (raw, shrunk):
                    edge = pred ^ ndi.binary_erosion(pred)
                    matched = np.sum(edge & ndi.binary_dilation(target_edge,structure=disk))
                    matched += np.sum(target_edge & ndi.binary_dilation(edge,structure=disk))
                    surface = matched/(edge.sum()+target_edge.sum())
                    region = 2*(pred & target).sum()/(pred.sum()+target.sum())
                    values.append((region,surface))
                s = row["scores"][index]
                np.testing.assert_allclose(values, [[s["region"],s["surface2"]],[s["smaller_region"],s["smaller_surface2"]]], atol=1e-12,rtol=0)
                changes.append(100*(values[1][1]-values[0][1]))
                observations += 1
            np.testing.assert_allclose(changes,row["surface_delta"],rtol=0,atol=1e-10)
            low,high = sorted(changes)
            direction = "tie involved" if min(map(abs,changes))<1e-9 else (
                "opposite" if low<0<high else ("both improve" if low>0 else "both worsen"))
            subsets = ["all", "apparent_same_target" if row["same_target"] else "flagged"]
            if row["same_target"] and row["localized"]:
                subsets.append("localized_same_target")
            for group in subsets:
                counts[group][direction] += 1
                for threshold in (1,2,5):
                    if low<=-threshold and high>=threshold:
                        counts[group][threshold] += 1
    for group,c in counts.items():
        primary = summary["subsets"][group]["surface_delta"]
        assert all(c[k]==n for k,n in primary["outcomes"].items())
        assert all(c[int(k)]==n for k,n in primary["reversal_min_points"].items())
    print(f"Verified {len(rows)} baseline hashes, {observations} paired-reference evaluations and all four primary result strata.")


if __name__=="__main__":
    check()

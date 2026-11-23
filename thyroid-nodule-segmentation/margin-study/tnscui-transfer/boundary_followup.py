"""Descriptive follow-up in every boundary record, including alternative references."""
import json
import zipfile

import numpy as np
import torch

from transfer import HERE, LOCAL, ROOT, file_hash, inference, save


def run():
    torch.set_num_threads(2)
    result = json.loads((HERE / "results.json").read_bytes())
    audit = LOCAL / "audit.json"
    primary_scores = LOCAL / "per-image.json"
    archive_path = LOCAL / "TNSC2020.zip"
    checkpoint = ROOT / "runs/source-fold0/gray/checkpoint.pt"
    for path, key in ((audit, "audit_sha256"), (primary_scores, "per_image_sha256"),
                      (archive_path, "archive_sha256"), (checkpoint, "checkpoint_sha256")):
        if file_hash(path) != result[key]:
            raise ValueError(f"Input changed: {path.name}")
    cases = [r for r in json.loads(audit.read_bytes())["rows"] if r["boundary"]]
    with zipfile.ZipFile(archive_path) as z:
        rows = inference(z, cases, np.array(result["fixed_fractions"]), checkpoint)
    primary = {r["id"]: r for r in json.loads(primary_scores.read_bytes())}
    common = [r for r in rows if r["id"] in primary]
    if not common or any(r != primary[r["id"]] for r in common):
        raise ValueError("Overlapping primary predictions do not match")
    change = np.array([100*(r["adaptive"]["dice"]-r["whole"]["dice"]) for r in rows])
    removed_tp = [r["whole"]["tp"]-r["adaptive"]["tp"] for r in rows]
    removed_fp = [r["whole"]["fp"]-r["adaptive"]["fp"] for r in rows]
    new_misses = sum(r["whole"]["tp"] > 0 and r["adaptive"]["tp"] == 0 for r in rows)
    if any(tp < 0 or tp > case["outside"] for tp, case in zip(removed_tp, cases, strict=True)):
        raise ValueError("Removed true positives exceed excluded reference")
    per_image = LOCAL / "boundary-scores.json"
    save(per_image, rows)
    summary = dict(records=len(cases), distinct_images=len({r["image_hash"] for r in cases}),
                   overlapping_primary_records=len(common), records_losing_true_positives=sum(tp>0 for tp in removed_tp),
                   distinct_images_losing_true_positives=len({r["image_hash"] for r,tp in zip(cases,removed_tp) if tp>0}),
                   removed_tp=sum(removed_tp), removed_fp=sum(removed_fp), new_misses=new_misses,
                   dice_improved=int(sum(change>0)), dice_worsened=int(sum(change<0)),
                   dice_unchanged=int(sum(change==0)), worst_change_points=float(change.min()),
                   baseline_mean_dice=float(np.mean([r["whole"]["dice"] for r in rows])),
                   adaptive_mean_dice=float(np.mean([r["adaptive"]["dice"] for r in rows])),
                   mean_change_points=float(change.mean()), primary_result_sha256=file_hash(HERE/"results.json"),
                   per_image_sha256=file_hash(per_image))
    save(HERE / "boundary-results.json", summary)
    print(summary)


if __name__ == "__main__":
    run()

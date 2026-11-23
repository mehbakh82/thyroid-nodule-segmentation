"""Create the public numeric tables from completed local development experiments."""
import argparse
import csv
import json
from pathlib import Path
from shutil import copyfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def write(path, records):
    path.write_text(json.dumps(records, separators=(",", ":"), allow_nan=False) + "\n")


def export(local, dest):
    dest.mkdir(exist_ok=True, parents=True)
    copyfile(local/"margin-study/per-image.json", dest/"tn3k-margin.json")
    copyfile(local/"tnui-control/per-image-scores.json", dest/"tnui-codec.json")
    for folder, filename, output in (
        ("tnui-location", "per-image.json", "tnui-localization"),
        ("tnui-location", "border-per-image.json", "tnui-border-discovery"),
        ("tnui-border-repeat", "per-image.json", "tnui-border-repeat"),
    ):
        records = read(local/folder/filename)
        order = {r["name"]:i for i,r in enumerate(read(local/folder/"sample.json")["rows"],1)}
        public = [dict(case=order[r["name"]], **{k:v for k,v in r.items() if k!="name"}) for r in records]
        write(dest/(output+".json"), public)

    audit = read(local/"tnscui-transfer/audit.json")["rows"]
    record_ids = {r["id"]:i for i,r in enumerate(audit,1)}
    scans = {key:i for i,key in enumerate(sorted({r["image_hash"] for r in audit}),1)}
    public = [dict(record=record_ids[r["id"]], scan=scans[r["image_hash"]],
                   **{k:v for k,v in r.items() if k not in {"id","image_hash","mask_hash"}}) for r in audit]
    write(dest/"tnscui-field.json", public)
    for filename, output in (("per-image.json","tnscui-model"),("boundary-scores.json","tnscui-boundary")):
        records = read(local/"tnscui-transfer"/filename)
        public = [dict(record=record_ids[r["id"]], **{k:v for k,v in r.items() if k!="id"}) for r in records]
        write(dest/(output+".json"), public)
    pairs = read(local/"paired-references/per-pair.json")
    write(dest/"paired-edit-scores.json",[{k:v for k,v in r.items() if k not in {"ids","image_hash","baseline_sha256"}}
                                          for r in pairs])
    for filename in ("paired-changes.csv","model-ranking.csv"):
        with (local/"paired-references"/filename).open(newline="") as inp, (dest/filename).open("w",newline="") as out:
            reader=csv.DictReader(inp)
            keys=[key for key in reader.fieldnames if key not in ("id_A","id_B")]
            writer=csv.DictWriter(out,fieldnames=keys,lineterminator="\n")
            writer.writeheader()
            writer.writerows({key:row[key] for key in keys} for row in reader)
    copyfile(local/"paired-references/model-ranking-predictions.npz",dest/"model-ranking-predictions.npz")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, default=HERE.parent/"local")
    p.add_argument("--out", type=Path, default=HERE/"case-results")
    args=p.parse_args()
    export(args.local,args.out)

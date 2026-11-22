"""Index a pinned DDTI mirror for image-level transfer testing before opening predictions."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pixels(archive, name, mode):
    with Image.open(io.BytesIO(archive.read(name))) as image:
        image=image.convert(mode)
        if image.size!=(256,256):
            raise ValueError(f'Unexpected dimensions for {name}')
        if mode=='L':
            counts=image.getcolors(256)
            if {v for _,v in counts}!={0,255}:
                raise ValueError(f'Unexpected label values for {name}')
        return sha(image.tobytes())


def manifest(path):
    groups=defaultdict(lambda:defaultdict(list))
    with ZipFile(path) as z:
        files=set(z.namelist())
        masks=sorted(n for n in files if '/label/' in n and n.endswith('.png'))
        for mask in masks:
            image=mask.replace('/label/','/img/')
            alias=image[:-4]+'.PNG.png'
            ih=pixels(z,image,'RGB')
            if pixels(z,alias,'RGB')!=ih:
                raise ValueError('Alias mismatch: '+alias)
            groups[ih][pixels(z,mask,'L')].append(PurePosixPath(mask).stem)
    records=[]
    for ih,targets in sorted(groups.items()):
        for mh,ids in sorted(targets.items()):
            records.append(dict(image_hash=ih,mask_hash=mh,ids=sorted(ids),source_rows=len(ids),
                                reference_weight='1/'+str(len(targets)),unique_references=len(targets)))
    return groups,records


def main(root):
    evidence=root.parent/'msc-readiness'/'thyroid-evidence'
    source=json.loads((evidence/'sources.json').read_bytes())['checked_files']['ddti.zip']
    archive=evidence/'local-data'/'ddti.zip'
    if sha(archive.read_bytes())!=source['sha256']:raise ValueError('DDTI release mismatch')
    groups,records=manifest(archive)
    for ih in groups:
        total=sum(Fraction(r['reference_weight']) for r in records if r['image_hash']==ih)
        assert total==1
    summary=dict(archive_sha256=source['sha256'],paired_rows=sum(r['source_rows'] for r in records),
                 unique_images=len(groups),unique_pairs=len(records),
                 single_reference_images=sum(len(m)==1 for m in groups.values()),
                 unique_reference_counts=dict(Counter(len(m) for m in groups.values())))
    assert (summary['paired_rows'],summary['unique_images'],summary['unique_pairs'],summary['single_reference_images'])==(637,630,634,626)
    serialized=(json.dumps(records,sort_keys=True,indent=2)+'\n').encode()
    out=root/'local'/'ddti-preflight'
    out.mkdir(parents=True,exist_ok=True)
    (out/'image-reference-map.json').write_bytes(serialized)
    summary['manifest_sha256']=sha(serialized)
    (root/'results'/'ddti-preflight.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)


if __name__=='__main__':main(Path(__file__).resolve().parent)

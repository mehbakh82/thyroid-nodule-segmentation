"""Fetch a fixed author-training sample. Verify each file against the pinned Git blob."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen

REPO='zxg3017/TNUI-2021-'
REV='5fa417629f581d82017eed535cdd62586717b669'
SALT='tnui-codec-v1/'


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'thyroid-dataset-preflight'}), timeout=35) as f:
                return f.read()
        except OSError:
            if attempt==2:raise
            sleep(2**attempt)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch('https://api.github.com/repos/'+REPO+'/git/trees/'+REV+'?recursive=1'))
    assert not tree['truncated']
    entries={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    prefix='thyroid_data/train/'
    names=[p.removeprefix(prefix+'masks/') for p in entries if p.startswith(prefix+'masks/') and p.endswith('.png')]
    assert len(names)==966 and all(prefix+'images/'+n in entries for n in names)
    chosen=sorted(names,key=lambda n:hashlib.sha256((SALT+n).encode()).digest())[:48]
    def one(name):
        record={'name':name}
        for role in ('images','masks'):
            key=prefix+role+'/'+name
            path=out/role/name
            b=path.read_bytes() if path.exists() else fetch('https://raw.githubusercontent.com/'+REPO+'/'+REV+'/'+key)
            digest=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
            assert digest==entries[key]['sha']
            path.parent.mkdir(exist_ok=True);path.write_bytes(b)
            record[role]=digest
        return record
    with ThreadPoolExecutor(max_workers=5) as ex:records=list(ex.map(one,chosen))
    dest=out/'sample.json'
    dest.write_text(json.dumps(dict(repo=REPO,revision=REV,partition='train',rows=records),indent=2)+'\n')
    print('Verified',len(records),'pairs out of',len(names),'training pairs.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out)

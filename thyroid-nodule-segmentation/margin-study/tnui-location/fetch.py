"""Fetch only the new diagnostic development sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

CONTROL=Path(__file__).resolve().parents[1]/'tnui-control'
sys.path.insert(0,str(CONTROL))
from fetch_sample import fetch,REPO,REV,SALT

PREFIX='thyroid_data/train/'
def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def select(names,salt):
    return sorted(names,key=lambda n:hashlib.sha256((salt+n).encode()).digest())

def run(out,previous,location_sample=None):
    old=json.loads(previous.read_bytes())
    assert old['revision']==REV and old['partition']=='train'
    out.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1'))
    assert not tree['truncated']
    files={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    names={Path(k).name for k in files if k.startswith(PREFIX+'masks/') and k.endswith('.png')}
    assert len(names)==966
    old_names={r['name'] for r in old['rows']}
    assert old_names==set(select(names,SALT)[:48])
    salt='tnui-location-v1/'
    chosen=select(names-old_names,salt)[:96]
    extra={}
    if location_sample is not None:
        first=json.loads(location_sample.read_bytes())
        assert first['revision']==REV and first['partition']=='train'
        assert set(chosen)=={r['name'] for r in first['rows']}
        salt='tnui-display-repeat-v1/'
        chosen=select(names-old_names-set(chosen),salt)[:48]
        extra['location_sample_sha256']=hashlib.sha256(location_sample.read_bytes()).hexdigest()
    def pair(name):
        row={'name':name}
        for kind in ('images','masks'):
            key=PREFIX+kind+'/'+name;dest=out/kind/name
            raw=dest.read_bytes() if dest.exists() else fetch(f'https://raw.githubusercontent.com/{REPO}/{REV}/{key}')
            assert blob(raw)==files[key],key
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True)
                tmp=dest.with_suffix('.part');tmp.write_bytes(raw);tmp.replace(dest)
            row[kind]=files[key]
        return row
    with ThreadPoolExecutor(max_workers=5) as pool: rows=list(pool.map(pair,chosen))
    receipt={'repo':REPO,'revision':REV,'partition':'train','salt':salt,**extra,
             'prior_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),'rows':rows}
    (out/'sample.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'Checked {2*len(rows)} blobs; {len(rows)} newly selected pairs.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('previous',type=Path)
    p.add_argument('--location-sample',type=Path)
    a=p.parse_args();run(a.out,a.previous,a.location_sample)

"""Complement the biomedical index with capped, relevance-ranked Crossref searches."""
import json
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from pathlib import Path

queries=['thyroid nodule ultrasound segmentation domain generalization external validation DDTI TN3K',
         'thyroid nodule ultrasound lesion segmentation boundary edge contrast marker caliper observer',
         'thyroid nodule ultrasound segmentation dataset cine public challenge benchmark']
if __name__=='__main__':
    records={};searches=[]
    for query in queries:
        params={'query.bibliographic':query,'rows':75,'filter':'until-pub-date:2026-11-21','select':'DOI,title,published,container-title,type'}
        url='https://api.crossref.org/works?'+urlencode(params)
        with urlopen(Request(url,headers={'User-Agent':'Research-scoping-review/1.0'}),timeout=65) as r:
            items=json.load(r)['message']['items']
        searches.append({'query':query,'retrieved':len(items)})
        for item in items:
            doi=item['DOI'].lower()
            rec=records.setdefault(doi,dict(doi=doi,title=next(iter(item.get('title',[])),''),year=item.get('published',{}).get('date-parts',[[None]])[0][0],type=item['type'],queries=[]))
            rec['queries'].append(query)
    d=Path(__file__).resolve().parent
    index=json.loads((d/'search-index.json').read_text())
    common=set(records)&{v['doi'] for v in index['records'] if v.get('doi')}
    (d/'crossref-index.json').write_text(json.dumps(dict(as_of='2026-11-21',cap=75,queries=searches,unique_dois=len(records),common_dois_with_europepmc=len(common),records=list(records.values())),ensure_ascii=False,indent=2)+'\n')
    print(len(records),len(common))

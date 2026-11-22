"""Snapshot a bounded, deduplicated Europe PMC scoping search. No clinical images used."""
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from time import sleep

BASE='https://www.ebi.ac.uk/europepmc/webservices/rest/search'
Q={
 'target':'(TITLE_ABS:thyroid AND TITLE_ABS:nodul* AND (TITLE_ABS:ultrasound OR TITLE_ABS:ultrasonograph*) AND (TITLE:segment* OR TITLE:delineat* OR TITLE:contour*))',
 'benchmarks':'((TITLE_ABS:TN3K OR TITLE_ABS:DDTI OR TITLE_ABS:ThyroidXL OR TITLE_ABS:TN5000 OR TITLE_ABS:"TN-SCUI") AND (TITLE_ABS:thyroid OR TITLE_ABS:nodul*))',
 'transport':'(TITLE_ABS:thyroid AND TITLE_ABS:nodul* AND TITLE_ABS:segment* AND (TITLE_ABS:"external validation" OR TITLE_ABS:"cross-dataset" OR TITLE_ABS:"domain generalization" OR TITLE_ABS:caliper* OR TITLE_ABS:marker OR TITLE_ABS:interobserver))',
 'resources':'(TITLE:thyroid AND (TITLE:ultrasound OR TITLE:ultrasonograph*) AND (TITLE:dataset OR TITLE:database OR TITLE:benchmark OR TITLE:"data set"))'}
def page(params):
    url=BASE+'?'+urlencode(params)
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'thyroid-research-scoping-review'}),timeout=45) as r:return json.load(r)
        except OSError:
            if attempt==2:raise
            sleep(2**attempt)
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    studies={};queries={}
    for k,q in Q.items():
        query=q+' AND FIRST_PDATE:[1900-01-01 TO 2026-11-21]'
        cur='*';count=0
        while True:
            data=page(dict(query=query,format='json',pageSize=1000,cursorMark=cur,resultType='lite'))
            batch=data['resultList']['result'];count+=len(batch)
            for r in batch:
                title=r.get('title','');doi=r.get('doi','').lower().strip()
                key=doi or ('pmid:'+str(r['pmid']) if r.get('pmid') else 'title:'+''.join(c for c in title.lower() if c.isalnum()))
                rec=studies.setdefault(key,{'title':title,'doi':doi,'pmid':r.get('pmid'),'year':r.get('pubYear'),'journal':r.get('journalTitle'),'queries':[]})
                if k not in rec['queries']:rec['queries'].append(k)
            nxt=data.get('nextCursorMark')
            if count>=data['hitCount'] or not batch or not nxt or nxt==cur:break
            cur=nxt
        queries[k]=dict(query=query,hits=data['hitCount'],retrieved=count)
    (root/'search-index.json').write_text(json.dumps(dict(as_of='2026-11-21',endpoint=BASE,queries=queries,unique=len(studies),records=sorted(studies.values(),key=lambda a:(a['year'] or '',a['title']))),ensure_ascii=False,indent=2)+'\n')
    print({k:v['retrieved'] for k,v in queries.items()},len(studies))

"""Preserve a blinded case assignment and any answers collected against it."""
import csv
import io
import json
import os
import tempfile


def create_once(path, content):
    """Publish a complete new file without replacing an existing one."""
    with tempfile.NamedTemporaryFile(dir=path.parent) as staged:
        staged.write(content.encode())
        staged.flush()
        try:
            os.link(staged.name,path)
        except FileExistsError:
            return False
    return True


def save_key(folder, assignment):
    path=folder/'key.json'
    if not path.exists() and (folder/'response-sheet.csv').exists():
        raise ValueError('An existing response sheet has no matching key; use a new review directory.')
    if not create_once(path,json.dumps(assignment,indent=2)+'\n'):
        if json.loads(path.read_bytes())!=assignment:
            raise ValueError('Different case assignment; use a new review directory.')


def prepare_sheet(path, assignment, appearance_first):
    fields=['target_clear','margin_visibility','calipers'] if appearance_first else []
    fields+=['preference','main_concern','consequential_correction_and_location']
    header=['case',*fields]
    buffer=io.StringIO(newline='')
    writer=csv.writer(buffer)
    writer.writerow(header)
    cases=[str(r['case']) for r in assignment]
    writer.writerows([[case,*(['']*len(fields))] for case in cases])
    if not create_once(path,buffer.getvalue()):
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=header or sorted(r['case'] for r in reader)!=sorted(cases):
                raise ValueError('Existing response sheet has a different schema or case set.')

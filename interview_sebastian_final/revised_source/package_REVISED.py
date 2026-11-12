#!/usr/bin/env python3
"""Bundle only the revision and its source/assets; leave the original ZIP alone."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parent

def main():
    required=['Sebastian_Research_Interview_Main_Only_REVISED.pdf',
              'Sebastian_Research_Interview_REVISED.pdf',
              'Sebastian_Research_Interview_REVISED.pptx',
              'Sebastian_Research_Interview_REVISED.tex',
              'Speaker_Notes_REVISED.md','Scientific_Changes.md','Final_Validation.md','README_REVISED.md']
    checks=json.loads((HERE/'validation/checks.json').read_text())
    assert not checks['failed'],checks['failed']
    manifest={name:{'bytes':(OUT/name).stat().st_size,
              'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()} for name in required}
    (HERE/'validation/revised_deliverable_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[OUT/name for name in required]+[OUT/'Sources_and_Evidence.md']
    for folder in [HERE,OUT/'rendered_REVISED']:
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/'V4.pdf',ROOT/'EULER_STABILITY_RESEARCH_AUDIT.md']
    target=OUT/'Sebastian_Interview_REVISED_Package.zip'
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
    with ZipFile(target) as z:
        assert z.testzip() is None
        count=len(z.namelist())
    print(json.dumps({'zip':str(target),'files':count,'bytes':target.stat().st_size,'CRC':'passed'}))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Finalize revised PDFs and render review images without touching originals."""
from pathlib import Path
import argparse,json,shutil,subprocess,tempfile
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
VALID=HERE/'validation'
RENDERED=OUT/'rendered_REVISED'
NAME='Sebastian_Research_Interview_REVISED'
MAIN='Sebastian_Research_Interview_Main_Only_REVISED'

def run(*args):
    return subprocess.run([str(x) for x in args],check=True,capture_output=True,text=True).stdout

def page(pdf,dest,width=1800):
    run('pdftoppm','-f',1,'-l',1,'-singlefile','-png','-scale-to-x',width,'-scale-to-y',-1,pdf,dest.with_suffix(''))

def sheet(paths,dest,first=1):
    rows=(len(paths)+2)//3
    im=Image.new('RGB',(1500,314*rows),'#e8e8e8');draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf',17)
    for i,p in enumerate(paths):
        small=Image.open(p).convert('RGB');small.thumbnail((480,270))
        x,y=(i%3)*500+10,(i//3)*314+10;im.paste(small,(x,y))
        label=f'Slide {i+first:02d}' if i+first<=9 else f'Backup {chr(65+i+first-10)}'
        draw.text((x,y+277),label,fill='#242424',font=font)
    im.save(dest)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-render',type=Path);a=p.parse_args()
    VALID.mkdir(exist_ok=True);RENDERED.mkdir(exist_ok=True)
    pdf=OUT/f'{NAME}.pdf';shutil.copy2(pdf,VALID/'Beamer_Render.pdf')
    with tempfile.TemporaryDirectory(prefix='kit-revised-') as tmp:
        tmp=Path(tmp);run('pdfseparate',pdf,tmp/'page-%02d.pdf')
        pages=[HERE/'assets/V4_landing-1.pdf']+[tmp/f'page-{i:02d}.pdf' for i in range(2,13)]
        run('pdfunite',*pages,tmp/'full.pdf');shutil.copy2(tmp/'full.pdf',pdf)
        run('pdfunite',*pages[:9],OUT/f'{MAIN}.pdf')
    run('pdftoppm','-png','-scale-to-x',1500,'-scale-to-y',-1,pdf,RENDERED/'slide')
    sheet([RENDERED/f'slide-{i:02d}.png' for i in range(1,10)],RENDERED/'Main_Contact_Sheet.png')
    sheet([RENDERED/f'slide-{i:02d}.png' for i in range(10,13)],RENDERED/'Backup_Contact_Sheet.png',10)
    original_v4=OUT.parent/'V4.pdf'
    page(original_v4 if original_v4.exists() else HERE/'assets/V4_landing-1.pdf',VALID/'V4_Landing.png')
    original_main=OUT/'Sebastian_Research_Interview_Main_Only.pdf'
    page(original_main if original_main.exists() else HERE/'assets/V4_landing-1.pdf',VALID/'Original_Landing.png')
    page(OUT/f'{MAIN}.pdf',VALID/'Final_PDF_Landing.png')
    if a.native_render:
        native=VALID/'Native_PowerPoint_Render.pdf';shutil.copy2(a.native_render,native)
        page(native,VALID/'PowerPoint_Landing.png')
        native_dir=VALID/'native_rendered';native_dir.mkdir(exist_ok=True)
        run('pdftoppm','-png','-scale-to-x',1500,'-scale-to-y',-1,native,native_dir/'slide')
        sheet([native_dir/f'slide-{i:02d}.png' for i in range(1,10)],VALID/'PowerPoint_Main_Contact_Sheet.png')
    compare=VALID/'comparisons';compare.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf',23)
    for i in range(1,10):
        old_path=OUT/f'rendered/slide-{i:02d}.png'
        if not old_path.exists():continue
        old=Image.open(old_path).convert('RGB')
        new=Image.open(RENDERED/f'slide-{i:02d}.png').convert('RGB')
        im=Image.new('RGB',(old.width+new.width,old.height+45),'#e8e8e8')
        im.paste(old,(0,45));im.paste(new,(old.width,45));draw=ImageDraw.Draw(im)
        draw.text((20,10),f'ORIGINAL — slide {i}',font=font,fill='#242424')
        draw.text((old.width+20,10),f'REVISED — slide {i}',font=font,fill='#242424')
        im.save(compare/f'slide-{i:02d}.png')
    print('Revised full PDF: 12 pages; main PDF: 9. Rendered all slides and original/revised comparisons.')

if __name__=='__main__':main()

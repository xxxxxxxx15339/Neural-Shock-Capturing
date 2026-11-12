#!/usr/bin/env python3
"""Preserve the original vector landing page; render and package review evidence.

Run after compiling Sebastian_Research_Interview.tex. Requires Poppler and Pillow.
Optional --native-render points to a LibreOffice PDF including hidden slides.
No experimental code or data are executed or changed.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
NAME = 'Sebastian_Research_Interview'

def run(*args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True).stdout

def page_png(pdf, destination, page=1, width=1800):
    run('pdftoppm', '-f', page, '-l', page, '-singlefile', '-png',
        '-scale-to-x', width, '-scale-to-y', '-1', pdf, destination.with_suffix(''))

def contact_sheet(paths, destination, titles):
    cell_w, cell_h = 500, 314
    sheet = Image.new('RGB', (3*cell_w, 4*cell_h), '#e8e8e8')
    font = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf', 16)
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        im = Image.open(path).convert('RGB'); im.thumbnail((480, 270))
        x, y = (index % 3)*cell_w+10, (index//3)*cell_h+10
        sheet.paste(im, (x, y))
        draw.text((x, y+276), f'{index+1:02d}  {titles[str(index+1)]["title"][:55]}', fill='#242424', font=font)
    sheet.save(destination)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-render', type=Path)
    args = p.parse_args()
    validation = HERE/'validation'; rendered = HERE/'rendered'
    validation.mkdir(exist_ok=True); rendered.mkdir(exist_ok=True)
    pdf = HERE/f'{NAME}.pdf'
    # Keep the actual compiled source output for reproducibility and inspection.
    shutil.copy2(pdf, validation/'Beamer_Render.pdf')
    with tempfile.TemporaryDirectory(prefix='kit-final-') as tmp:
        tmp = Path(tmp)
        run('pdfseparate', pdf, tmp/'body-%02d.pdf')
        pages = [HERE/'assets/V4_landing-1.pdf'] + [tmp/f'body-{i:02d}.pdf' for i in range(2,13)]
        run('pdfunite', *pages, tmp/'complete.pdf')
        shutil.copy2(tmp/'complete.pdf', pdf)
        run('pdfunite', *pages[:9], HERE/f'{NAME}_Main_Only.pdf')
    run('pdftoppm', '-png', '-scale-to-x', 1500, '-scale-to-y', '-1', pdf, rendered/'slide')
    notes = json.loads((HERE/'notes_content.json').read_text())
    contact_sheet([rendered/f'slide-{i:02d}.png' for i in range(1,13)], rendered/'Contact_Sheet.png', notes)
    original = HERE.parent/'V4.pdf'
    page_png(original if original.exists() else HERE/'assets/V4_landing-1.pdf', validation/'V4_Landing.png')
    page_png(pdf, validation/'Final_PDF_Landing.png')
    if args.native_render:
        native = validation/'Native_PowerPoint_Render.pdf'
        shutil.copy2(args.native_render.resolve(), native)
        page_png(native, validation/'PowerPoint_Landing.png')
        native_dir = validation/'native_rendered'; native_dir.mkdir(exist_ok=True)
        run('pdftoppm', '-png', '-scale-to-x', 1500, '-scale-to-y', '-1', native, native_dir/'slide')
        contact_sheet([native_dir/f'slide-{i:02d}.png' for i in range(1,13)], validation/'PowerPoint_Contact_Sheet.png', notes)
    print('Final PDF: 12 pages. Main-only PDF: 9 pages. Original vector cover retained; all pages rendered.')

if __name__ == '__main__':
    main()

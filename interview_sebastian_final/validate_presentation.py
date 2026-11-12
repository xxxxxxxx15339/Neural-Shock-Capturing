#!/usr/bin/env python3
"""Read-only presentation/data checks; writes validation reports only.

Run after finalize_presentation.py. No solver, training or external service runs.
The manual visual and scientific review is documented in Presentation_Validation.md.
"""
from pathlib import Path
from zipfile import ZipFile
import csv
import hashlib
import json
import math
import posixpath
import re
import subprocess
import sys
import numpy as np
from PIL import Image
from lxml import etree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALID = HERE/'validation'
NAME = 'Sebastian_Research_Interview'
NS = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
checks = []

def check(label, passed, details=None):
    checks.append({'check':label, 'passed':bool(passed), 'details':details})

def load(name):
    return json.loads((HERE/name).read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def page_count(p):
    info = subprocess.check_output(['pdfinfo', str(p)], text=True)
    return int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])

def near(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-15)

def pixel_comparison(a,b):
    aa = np.array(Image.open(a).convert('RGB')); bb = np.array(Image.open(b).convert('RGB'))
    if aa.shape != bb.shape:
        return {'same_dimensions':False, 'a':list(aa.shape), 'b':list(bb.shape)}
    diff = np.abs(aa.astype(np.int16)-bb.astype(np.int16))
    return {'same_dimensions':True, 'width':aa.shape[1], 'height':aa.shape[0],
            'changed_pixels':int(np.any(diff,axis=2).sum()),
            'mean_absolute_channel_error':float(diff.mean()),
            'max_channel_error':int(diff.max())}

def main():
    originals = load('validation/input_hashes.json')
    for path, expected in originals.items():
        p=ROOT/path
        check('Original input unchanged: '+path, p.exists() and digest(p)==expected)
    check('Portable editable V4 template is an exact copy',
          digest(HERE/'assets/V4_editable_template.pptx') == originals['Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pptx'])
    for file,n in [(f'{NAME}.pdf',12),(f'{NAME}_Main_Only.pdf',9),('validation/Native_PowerPoint_Render.pdf',12)]:
        count=page_count(HERE/file);check('PDF page count: '+file, count==n, count)
    for label,path in [('Final PDF','Final_PDF_Landing.png'),('Native PPTX via LibreOffice','PowerPoint_Landing.png')]:
        result=pixel_comparison(VALID/'V4_Landing.png',VALID/path)
        check(label+' landing identical to original V4 first page',
              result.get('same_dimensions') and result.get('changed_pixels')==0,result)

    inventory=[]
    with ZipFile(HERE/f'{NAME}.pptx') as z, ZipFile(HERE/'assets/V4_editable_template.pptx') as original:
        files=set(z.namelist());slide_names=[f'ppt/slides/slide{i}.xml' for i in range(1,13)]
        actual=[n for n in files if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        check('PPTX contains exactly 12 slide XML parts',set(actual)==set(slide_names),len(actual))
        check('Original editable first-slide XML unchanged',z.read(slide_names[0])==original.read(slide_names[0]))
        rel=ET.fromstring(original.read('ppt/slides/_rels/slide1.xml.rels'))
        image_paths=[posixpath.normpath(posixpath.join('ppt/slides',r.get('Target'))) for r in rel if r.get('Type').endswith('/image')]
        check('All original first-slide image/equation assets byte-identical',
              all(z.read(n)==original.read(n) for n in image_paths),image_paths)
        missing=[]; dangling=[]; xml_count=0
        for f in sorted(files):
            if f.endswith(('.xml','.rels')):
                ET.fromstring(z.read(f));xml_count+=1
            if not f.endswith('.rels'):continue
            root=ET.fromstring(z.read(f));parent=posixpath.dirname(posixpath.dirname(f));ids={}
            for r in root:
                ids[r.get('Id')]=r.get('Target')
                if r.get('TargetMode')=='External':continue
                target=posixpath.normpath(posixpath.join(parent,r.get('Target'))).lstrip('/')
                if target not in files:missing.append([f,target])
            source=posixpath.join(parent,posixpath.basename(f)[:-5])
            if source in files:
                for node in ET.fromstring(z.read(source)).iter():
                    for key,value in node.attrib.items():
                        if key.startswith('{'+NS['r']+'}') and value not in ids:dangling.append([source,key,value])
        check('All OOXML parses and package relationships resolve',not missing and not dangling,
              {'xml_parts':xml_count,'missing':missing,'dangling':dangling})
        notes=load('notes_content.json')
        for i,path in enumerate(slide_names,1):
            root=ET.fromstring(z.read(path))
            text_nodes=root.xpath('//p:sp[.//a:t]',namespaces=NS)
            shape_count=len(root.xpath('//p:sp',namespaces=NS));pictures=len(root.xpath('//p:pic',namespaces=NS))
            hidden=root.get('show')=='0'
            inventory.append({'slide':i,'native_text_shapes':len(text_nodes),'native_shapes':shape_count,'pictures':pictures,'hidden':hidden})
            check(f'Slide {i}: main visible / backup hidden',hidden==(i>9))
            note=ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml'))
            text='\n'.join(note.xpath('//a:t/text()',namespaces=NS))
            # OOXML stores blank paragraphs as empty a:t nodes; XPath text() drops
            # those empty nodes. Compare words while preserving their order/content.
            normalized=' '.join(text.split())
            check(f'Slide {i}: complete embedded script and Q&A',
                  ' '.join(notes[str(i)]['script'].split()) in normalized and
                  all(' '.join(q['a'].split()) in normalized for q in notes[str(i)]['qa']))
    (VALID/'object_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    check('Each slide contains native editable text',all(row['native_text_shapes']>=6 for row in inventory),inventory)

    notes=load('notes_content.json');seconds=sum(notes[str(i)]['seconds'] for i in range(1,10))
    words=sum(len(notes[str(i)]['script'].split()) for i in range(1,10))
    check('Main-slide estimated speaking time is 12–15 minutes',720<=seconds<=900,
          {'seconds':seconds,'minutes':seconds/60,'spoken_script_words':words,'words_per_minute':words/(seconds/60),'timing_is_estimate':True})
    check('Every main slide has a script, terms, preparation and 1–2 Q&A',
          all(notes[str(i)]['script'] and notes[str(i)]['terms'] and notes[str(i)]['preparation'] and 1<=len(notes[str(i)]['qa'])<=2 for i in range(1,10)))
    eq=load('validation/equation_inventory.json')
    main_min=min(e['effective_point_size'] for e in eq if e['slide']<=9)
    backup_min=min(e['effective_point_size'] for e in eq if e['slide']>9)
    check('Equations readable in source inventory',main_min>=21 and backup_min>=19,
          {'main_minimum_pt':main_min,'backup_minimum_pt':backup_min,'equation_instances':len(eq)})
    log=(HERE/f'{NAME}.log').read_text()
    check('TeX: no overfull boxes or missing glyphs', 'Overfull' not in log and 'Missing character' not in log,
          {'underfull_warnings':log.count('Underfull'),'retained_warning_types':['underfull fixed footer boxes','font path portability','original PDF version/tag handling']})
    layout=load('slide_layout.json');outside=[];missing_assets=[]
    for number,objects in layout.items():
        for ob in objects:
            if ob['x']<0 or ob['y']<0 or ob['x']+ob['w']>13.334 or ob['y']+ob['h']>7.501:outside.append([number,ob])
            if ob['type']=='image' and not (HERE/ob['path']).exists():missing_assets.append(ob['path'])
    check('Declared slide objects inside canvas; image assets present',not outside and not missing_assets,
          {'outside':outside,'missing':missing_assets})
    check('All 12 PDF slides and all 12 native-PPTX slides rendered',
          all((HERE/f'rendered/slide-{i:02d}.png').exists() and (VALID/f'native_rendered/slide-{i:02d}.png').exists() for i in range(1,13)))

    # Numerical values are checked against retained experiment artifacts, not recomputed simulations.
    val=next(r for r in load('assets/full_validation_summary.json')['summary'] if r['physics']=='euler' and r['mode']=='soft' and r['strength']==.25)
    check('V2 validation: 307, no failures, macro L1 0.94794',val['trajectories']==307 and val['failures']==0 and near(val['macro_relative_l1'],.9479383302719695),
          {k:val[k] for k in ['trajectories','failures','macro_relative_l1']})
    data=load('assets/independent_benchmark.json')['results']
    def row(case,method):return next(r for r in data if r['case']==case and r['cells']==192 and r['method']==method)
    a=row('independent-sod','teno5');b=row('independent-sod','v2-selected-soft')
    ratio=b['primary_error']/a['primary_error']
    check('Independent Sod: L1 improvement, higher TV, density undershoot',
          near(ratio,.9572500681506942) and near(a['total_variation'][0],.8774663254486642)
          and near(b['total_variation'][0],.8797561899875133) and near(b['undershoot'][0],.0003506854567733031),
          {'l1_ratio':ratio,'reduction_percent':100*(1-ratio),'teno5_density_TV':a['total_variation'][0],
           'selected_density_TV':b['total_variation'][0],'selected_density_undershoot':b['undershoot'][0]})
    contact=row('independent-contact','v2-selected-soft')['primary_error']/row('independent-contact','teno5')['primary_error']
    check('Independent contact regression',near(contact,1.0687408292618075),contact)
    wall=load('assets/wall_benchmark.json')['results'];wall_rows=[]
    for n in [128,256]:
        pair={m:next(r for r in wall if r['cells']==n and r['method']==m)['diagnostics'] for m in ['teno5','v2-selected-soft']}
        wall_rows.append({'N':n,'teno5_finite':pair['teno5']['finite'],'selected_finite':pair['v2-selected-soft']['finite']})
    check('Closed-wall blast: selected fails N128/N256, TENO5 finite',all(r['teno5_finite'] and not r['selected_finite'] for r in wall_rows),wall_rows)
    r=next(r for r in load('assets/repair_budget.json')['results'] if r['mode']=='soft' and r['strength']==.75 and r['repair_policy']=='historical-repair')['diagnostics']
    exchange=r['integrated_repair_exchange']
    check('One repaired stage; RK-weighted mass/energy sources',r['repaired_stage_rows']==1 and near(exchange[0],.0016111493654163188) and near(exchange[2],.0008888429518660344),exchange)
    orders={r['solver']:r['rows'][-1]['order_l1'] for r in load('assets/nonlinear_euler.json')['results'] if r['method']=='teno5'}
    check('Nonlinear Euler consistency orders 2.01 and 5.00',near(orders['historical-split-flux'],2.0119685376282312) and near(orders['experimental-fv-state-flux'],4.999925297394422),orders)
    probe=load('validation/audit_lightweight_probe.json')['first_invalid']['runs']
    weak=next(r for r in probe if r['mode']=='soft' and r['strength']==.25)
    strong=next(r for r in probe if r['mode']=='soft' and r['strength']==.75)
    check('Raw-stage V2 pressure failure: step 13 stage 1 versus 15 valid weak steps',
          weak['valid_steps']==15 and strong['first_invalid_step']==13 and strong['first_invalid_stage']==1 and strong['min_pressure']<0,{'weak':weak,'strong':strong})
    check('Oracle density failure supported by audit log','Non-positive Euler density at step 6.' in (VALID/'audit_oracle_probe.log').read_text())
    wlog=(VALID/'audit_weno7_probe.log').read_text();wdata=json.loads(wlog[wlog.index('{'):])
    check('WENO7: 10/100000 then 0; epsilon squared not underflow',wdata['rows']==100000 and [r['nonfinite_rows'] for r in wdata['runs']]==[10,0]
          and wdata['square_1e_6_float32']>wdata['float32_tiny'],wdata['runs'])
    traces=list(csv.DictReader((HERE/'assets/euler_instability_trace.csv').open()));failures={}
    for mult in [1.,1.5,2.,3.,5.]:
        rows=[r for r in traces if float(r['alpha_margin'])==mult]
        bad=[int(r['step']) for r in rows if r['finite']!='True']
        failures[str(mult)]={'first_nonfinite':min(bad) if bad else None,'last_density':rows[-1]['max_density']}
    check('LF sweep archived nonfinite indices 11/none/3/3/3',
          [r['first_nonfinite'] for r in failures.values()]==[11,None,3,3,3],failures)
    copies={'full_validation_summary.json':'docs/research_release/results/full-validation/summary.json',
            'repair_budget.json':'docs/research_release/results/repair/repair.json',
            'euler_instability_trace.csv':'models/euler_instability_trace.csv',
            'independent_benchmark.json':'docs/research_release/results/independent/benchmark.json',
            'wall_benchmark.json':'docs/research_release/results/wall-closure/benchmark.json',
            'nonlinear_euler.json':'docs/research_release/results/nonlinear-euler/nonlinear_euler.json',
            '04_euler_instability_original.png':'plots/04_euler_instability.png',
            'nonlinear_euler_consistency_original.png':'docs/portfolio/figures/nonlinear_euler_consistency.png',
            'euler_sod_original.png':'docs/portfolio/figures/euler_sod.png',
            'final_benchmark_original.png':'docs/portfolio/figures/final_benchmark.png',
            'sod192_selected.npz':'docs/research_release/results/independent/independent-sod-192/v2-selected-soft.npz',
            'sod192_teno5.npz':'docs/research_release/results/independent/independent-sod-192/teno5.npz',
            'sod192_reference.npz':'docs/research_release/results/independent/independent-sod-192/reference.npz'}
    for asset,source in copies.items():
        check('Experimental asset byte copy: '+asset,digest(HERE/'assets'/asset)==digest(ROOT/source))
    output={'checks_run':len(checks),'passed':sum(c['passed'] for c in checks),
            'failed':[c for c in checks if not c['passed']],'checks':checks}
    (VALID/'checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['checks_run','passed','failed']},indent=2))
    sys.exit(0 if not output['failed'] else 1)

if __name__=='__main__':
    main()

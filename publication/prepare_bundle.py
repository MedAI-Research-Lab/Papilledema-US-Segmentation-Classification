"""Maintainer export from the local study archive; never exports clinical pixels.

All writes are confined to this publication directory. This command is not needed
to replay the public results. Original sources remain unchanged.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import zipfile
from lxml import etree as E

HERE = Path(__file__).resolve().parent
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def text(el):
    return ''.join(el.xpath('.//w:t/text()', namespaces=NS))

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def selected_functions(source, names):
    tree = ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names)

def sanitize(source):
    # Runtime paths are configuration, not study evidence. Font substitutions are
    # explicit: exact reproduction requires the original locally licensed Arial.
    source = source.replace('from pathlib import Path', 'from pathlib import Path\nimport os\nimport shutil')
    source = re.sub(r'Path\("C:/Users/[^\"]+/pdftoppm.exe"\)', 'Path(os.environ.get("PUBLICATION_PDFTOPPM", shutil.which("pdftoppm") or "pdftoppm"))', source)
    source = re.sub(r'Path\("C:/Users/[^\"]+/node.exe"\)', 'Path(os.environ.get("PUBLICATION_NODE", shutil.which("node") or "node"))', source)
    source = re.sub(r'Path\("C:/Users/[^\"]+/sharp"\)', 'Path(os.environ.get("PUBLICATION_SHARP", "sharp"))', source)
    source = source.replace('"C:/Windows/Fonts/arial.ttf"', 'os.environ.get("PUBLICATION_ARIAL_REGULAR", "Arial.ttf")')
    source = source.replace('"C:/Windows/Fonts/arialbd.ttf"', 'os.environ.get("PUBLICATION_ARIAL_BOLD", "Arial-Bold.ttf")')
    source = '\n'.join(line for line in source.splitlines() if not ('sys.path.insert' in line and '.figure_deps' in line))+'\n'
    assert 'C:/Users/' not in source and 'C:/Windows/' not in source
    return source

def archive(study):
    mapping = {
      'classification.py':'publication_revision_20260926/classification/build_classification_figures.py',
      'classification_eye_validator.py':'publication_revision_20260926_v3/classification/build_classification_figures.py',
      'restored_eye_confusions.py':'publication_revision_20260926_restored_complete/classification_eye/build_eye_confusions.py',
      'segmentation_metrics.py':'publication_revision_20260926/segmentation_metrics/build_segmentation_figures.py',
      'qualitative.py':'publication_revision_20260926/segmentation/build_segmentation_panels.py',
      'methods_flowchart.py':'methods_audit_20260925/build_flowchart_2_visual.py',
      'binary_curves.py':'publication_complete_supplements_20260928/work/build_binary_supp_figures.py',
    }
    records=[]
    for name, rel in mapping.items():
        p=study/rel; src=p.read_text('utf-8-sig'); target=HERE/'original_generators'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        adapted=sanitize(src)
        # Make the historical dependency closure importable from this archive.
        if name=='classification_eye_validator.py':
            adapted=adapted.replace("old = ROOT/'publication_revision_20260926/classification/build_classification_figures.py'", "old = Path(__file__).with_name('classification.py')")
        if name=='restored_eye_confusions.py':
            adapted=adapted.replace("V3 / 'build_classification_figures.py'", "Path(__file__).with_name('classification_eye_validator.py')")
        target.write_text(adapted,encoding='utf-8')
        records.append({'original_source':rel,'original_sha256':sha(p),'packaged_source':str(target.relative_to(HERE)).replace('\\','/'),'packaged_sha256':sha(target),'adaptation':'Runtime paths parameterized; historical data dependencies preserved; .figure_deps path removed. Run through publication CLI.'})
    rel='threeclass_q1_manuscript_20260922/build_manuscript.py';p=study/rel;src=p.read_text('utf-8-sig')
    names={'bv','bms','bseg','bseg_level','v','ms','binary_eye_figure','segmentation_overall_figure','threeclass_patient_figure'}
    extracted='from pathlib import Path\nfrom PIL import Image, ImageDraw, ImageFont\n\n'+selected_functions(src,names)+'\n'
    target=HERE/'original_generators/legacy_summaries.py';target.write_text(sanitize(extracted),encoding='utf-8')
    records.append({'original_source':rel,'original_sha256':sha(p),'packaged_source':str(target.relative_to(HERE)).replace('\\','/'),'packaged_sha256':sha(target),'adaptation':'Exact figure/helper function extraction only; historical manuscript prose and all document-building functions excluded; local font paths parameterized.','functions':sorted(names)})
    dump(HERE/'provenance/generator_sources.json',records)

def tables(study):
    records=[]
    image_sources={}
    for folder in ['methods_audit_20260925','threeclass_q1_manuscript_20260922','publication_revision_20260926/classification','publication_revision_20260926/segmentation','publication_revision_20260926/segmentation_metrics','publication_revision_20260926_restored_complete/classification_eye','publication_complete_supplements_20260928/work/new_figures']:
        for image in (study/folder).glob('*.png'):
            image_sources.setdefault(sha(image),[]).append(image)
    for edition in (3000,5000):
        out=[];figures=[]
        for kind in ('Manuscript','Supplementary'):
            rel=f'publication_natural_20260928/under_{edition}/{kind}_{edition}.docx';p=study/rel
            with zipfile.ZipFile(p) as z:
                body=E.fromstring(z.read('word/document.xml')).find('w:body',NS)
                rels={e.get('Id'):e.get('Target') for e in E.fromstring(z.read('word/_rels/document.xml.rels'))}
                children=list(body)
                for i,el in enumerate(children):
                    for blip in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
                        rid=blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        digest=hashlib.sha256(z.read('word/'+rels[rid])).hexdigest()
                        matches=image_sources[digest]
                        chosen=matches[-1]
                        figures.append({'section':'main' if kind=='Manuscript' else 'supplement','caption':text(children[i-1]),'note':text(children[i+1]),'original_file':chosen.name,'sha256':digest,'source':str(chosen.relative_to(study)).replace('\\','/'),'requires_private_images':'segmentation' in chosen.parts and 'segmentation_metrics' not in chosen.parts})
            children=list(body)
            for i,el in enumerate(children):
                if el.tag!='{'+NS['w']+'}tbl':continue
                caption=text(children[i-1]);assert re.match(r'(?:Supplementary )?Table S?\d+\.',caption),caption
                rows=[[text(c) for c in row.findall('w:tc',NS)] for row in el.findall('w:tr',NS)]
                note=text(children[i+1]) if i+1<len(children) and text(children[i+1]).startswith('Note.') else ''
                out.append({'section':'main' if kind=='Manuscript' else 'supplement','number':re.search(r'Table (S?\d+)\.',caption)[1],'caption':caption,'rows':rows,'note':note})
            records.append({'source':rel,'sha256':sha(p),'operation':'Only table cell text, captions and table notes exported; no OOXML, media, or other manuscript prose.'})
        assert len(out)==31 and sum(r['section']=='main' for r in out)==5
        # Clinical-source narrative is supplied only from an authorized local
        # document, never from the public release. Enforce an explicit allowlist.
        clinical_number='S1' if edition==3000 else 'S2'
        public_numbers={str(i) for i in range(1,6)}|{f'S{i}' for i in range(1,27) if f'S{i}'!=clinical_number}
        public=[r for r in out if r['number'] in public_numbers]
        assert len(public)==30 and len(public_numbers)==30
        dump(HERE/f'tables/edition_{edition}.json',public)
        dump(HERE/f'tables/local_table_{edition}.json',{
            'section':'supplement','number':clinical_number,
            'source_option':'--supplement-docx',
            'source_sha256':sha(study/f'publication_natural_20260928/under_{edition}/Supplementary_{edition}.docx'),
            'role':'Clinical provenance and acquisition; rendered from an authorized local document.'})
        assert len(figures)==68 and sum(f['requires_private_images'] for f in figures)==6
        dump(HERE/f'provenance/figures_{edition}.json',figures)
        source=study/f'publication_editorial_20260928/work/numbering_{edition}.json'
        shutil.copy2(source,HERE/f'tables/numbering_{edition}.json')
    dump(HERE/'provenance/table_sources.json',records)

def aggregates(study):
    source=study/'publication_natural_20260928/under_3000/data'
    mapping={p.name:p for p in source.glob('*.csv')}
    b=study/'strict_roi_results_4model_v1_2_0/summary';t=study/'threeclass_roi_results_4model_v1_0_0/tables'
    for name in ['segmentation_per_seed.csv','segmentation_five_seed_mean_sd.csv','classification_primary_model_specific_per_seed.csv','classification_primary_model_specific_five_seed_mean_sd.csv']:
        mapping['binary_'+name]=b/name
    for name in ['classification_five_seed_mean_sd.csv','segmentation_per_seed_class.csv','segmentation_five_seed_class.csv','uncertainty_metrics.csv','paired_primary_patient_cluster_bootstrap_holm.csv','risk_coverage.csv','patient_metrics.csv','classwise_metrics.csv']:
        mapping['threeclass_'+name]=t/name
    mapping['discrimination_curve_points.csv']=study/'publication_revision_20260926/classification/discrimination_curve_points.csv'
    evidence=[]
    forbidden={'patient_id','case_id','frame_id','eye_id','image_path','mask_path','audit_path','unit_index'}
    dest=HERE/'data';dest.mkdir(parents=True,exist_ok=True)
    for name,p in mapping.items():
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);cols=reader.fieldnames;rows=list(reader)
        assert not forbidden.intersection(cols),(p,cols)
        assert not any(re.search(r'(?:[A-Z]:[\\/]|/Users/|/home/)',str(v)) for row in rows for v in row.values()),p
        shutil.copy2(p,dest/name)
        evidence.append({'file':name,'source':str(p.relative_to(study)).replace('\\','/'),'sha256':sha(p),'rows':len(rows),'columns':cols})
    dump(HERE/'provenance/aggregate_sources.json',evidence)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--study-root',type=Path,required=True);args=ap.parse_args()
    archive(args.study_root.resolve());tables(args.study_root.resolve());aggregates(args.study_root.resolve())
    print(json.dumps({'public_table_records':60,'clinical_tables_local_only':2,'clinical_images_exported':0,'participant_rows_exported':0}))

if __name__=='__main__':main()

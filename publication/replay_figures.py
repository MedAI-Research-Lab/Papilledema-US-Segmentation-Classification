"""Replay publication figures from public aggregate coordinates, not predictions.

This is a release adapter, distinct from the originally executed generators.
Original plotting geometry, fonts, axes, seed coding and statistics are retained.
Six ultrasound example panels are available only through qualitative_local.py.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
sys.dont_write_bytecode=True

def module(name):
    spec=importlib.util.spec_from_file_location('publication_'+name,HERE/'original_generators'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def replace_once(value,old,new):
    assert value.count(old)==1,old[:100]
    return value.replace(old,new,1)

def install(mod,name,changes):
    value=inspect.getsource(getattr(mod,name))
    for old,new in changes:value=replace_once(value,old,new)
    exec(compile(value,'<public aggregate replay>','exec'),mod.__dict__)

def classification(out):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.rcdefaults();m=module('classification');m.OUT=out
    cells=pd.read_csv(DATA/'confusion_cells_by_seed.csv')
    for task in ['binary','threeclass']:
        names=['Normal','Abnormal'] if task=='binary' else m.CLASS_NAMES
        for model in m.MODELS:
            for seed in m.SEEDS:
                g=cells.query('task == @task and model == @model and seed == @seed')
                matrix=g.pivot(index='true_class',columns='predicted_outcome',values='count').reindex(index=names,columns=names+['Abstain']).to_numpy(int)
                assert matrix.sum()==36
                assert np.array_equal(matrix.sum(axis=1),[20,16] if task=='binary' else [20,8,8])
                m.COUNTS[task,model,seed]=matrix
    install(m,'confusion_figures',[("level = 'eye' if task == 'binary' else 'patient'","level = 'eye'"),('Direct three-class classification: patient-level confusion','Direct three-class classification: eye-level confusion')])
    m.confusion_figures()
    # Check mean/sample-SD values against the final published aggregate records.
    saved=pd.read_csv(DATA/'confusion_mean_sd.csv')
    for rec in m.FIGURES:
        if rec['category']!='main_confusion':continue
        for model in m.MODELS:
            g=saved[(saved.task==rec['task'])&(saved.model==model)]
            for row in g.itertuples():
                assert np.isclose(rec['means'][model][row.true_label][row.predicted_label],row.mean_percentage)
                assert np.isclose(rec['sample_sds'][model][row.true_label][row.predicted_label],row.sample_sd_percentage_points)
    m.PUBLIC_POINTS=pd.read_csv(DATA/'discrimination_curve_points.csv')
    m.PUBLIC_STATS=pd.read_csv(DATA/'Supplementary_Data_S1.csv')
    value=inspect.getsource(m.discrimination)
    start=value.index('                    pred = PREDICTIONS')
    end=value.index('                    axes[0].plot',start)
    value=value[:start]+'''                    group = PUBLIC_POINTS.query('task == @task and model == @model and seed == @seed and `class` == @cname')
                    roc = group.query("curve == 'ROC'").sort_values('point')
                    pr = group.query("curve == 'PR'").sort_values('point')
                    fpr, tpr = roc.x.to_numpy(), roc.y.to_numpy()
                    recall, precision = pr.x.to_numpy(), pr.y.to_numpy()
                    stats = PUBLIC_STATS.query('task == @task and model == @model and seed == @seed and `class` == @cname').iloc[0]
                    auroc, ap = stats.auroc, stats.average_precision
                    assert len(roc) > 1 and len(pr) > 1
'''+value[end:]
    start=value.index('                    CURVE_STATS.append')
    end=value.index('                axes[0].plot([0,1]',start)
    value=value[:start]+value[end:]
    exec(compile(value,'<aggregate ROC PR replay>','exec'),m.__dict__);m.discrimination(None,None)
    m.PUBLIC_RELIABILITY=pd.read_csv(DATA/'reliability_points.csv')
    value=inspect.getsource(m.reliability_and_risk)
    start=value.index("                pred = PREDICTIONS")
    end=value.index('                ax.plot(xs,ys',start)
    value=value[:start]+'''                cname=CLASS_NAMES[c]
                group = PUBLIC_RELIABILITY.query('model == @model and seed == @seed and `class` == @cname').sort_values('bin')
                xs, ys = group.mean_probability.to_list(), group.observed_frequency.to_list()
                assert (group.n > 0).all()
'''+value[end:]
    start=value.index("            pred = PREDICTIONS")
    end=value.index('            ax.step(',start)
    value=value[:start]+'''            check(f'risk_denominator_{model}_{seed}', len(group)==18 and np.allclose(group.coverage,np.arange(1,19)/18))
'''+value[end:]
    exec(compile(value,'<aggregate reliability risk replay>','exec'),m.__dict__)
    risk=pd.read_csv(DATA/'threeclass_risk_coverage.csv').query("classifier_strategy == 'model_specific'")
    m.reliability_and_risk({'risk_coverage.csv':risk})
    return {'figures':len(m.FIGURES),'numerical_checks':len(m.TESTS)}

def segmentation(out):
    import matplotlib.pyplot as plt
    plt.rcdefaults();m=module('segmentation_metrics');m.OUT=out
    m.CLASS_SOURCE=DATA/'threeclass_segmentation_per_seed_class.csv'
    m.POST_SOURCE=DATA/'binary_segmentation_per_seed.csv';m.main()
    return {'figures':8,'input_frames_per_split':252}

def legacy(out):
    import pandas as pd
    m=module('legacy_summaries');m.OUT=out
    m.MODELS=['yolo26','vit_method2','emcad','sam2_unet'];m.LABEL=dict(zip(m.MODELS,['YOLO26','ViT Method2','EMCAD','SAM2-U-Net']))
    def rows(name):return pd.read_csv(DATA/name).to_dict('records')
    m.BINARY_MEANS={(r['model'],r['classifier_strategy'],r['level'],r['scope'],r['arm']):r for r in rows('binary_probability_quality_mean_sd.csv')}
    m.BINARY_SEG={(r['model'],r['classifier_strategy'],r['level'],r['scope']):r for r in rows('binary_segmentation_five_seed_mean_sd.csv')}
    m.MEANS={(r['model'],r['strategy'],r['probability_state'],r['level'],r['metric']):r for r in rows('threeclass_classification_five_seed_mean_sd.csv')}
    m.segmentation_overall_figure();m.binary_eye_figure();m.threeclass_patient_figure()
    return {'figures':3,'aggregation':'saved five-seed mean and sample SD'}

def binary(out):
    import csv
    m=module('binary_curves');m.OUT=out;m.ROOT=HERE
    with (DATA/'binary_curve_coordinates.csv').open(encoding='utf-8-sig',newline='') as f:m.PUBLIC_ROWS=list(csv.DictReader(f))
    value=inspect.getsource(m.main)
    start=value.index('            folder = RUNS')
    end=value.index('            seed_audit =',start)
    value=value[:start]+'''            selected_cal = [{**r, 'lower':r['bin_lower'], 'upper':r['bin_upper']} for r in PUBLIC_ROWS if r['model']==model and int(r['seed'])==seed and r['bin']!='']
'''+value[end:]
    old='csv_rows(risk_path)';new="[r for r in PUBLIC_ROWS if r['model']==model and int(r['seed'])==seed and r['rank']!='']"
    value=replace_once(value,old,new)
    # Public reliability coordinates omit empty bins. Validate actual bin IDs
    # and original bounds without inventing observations or plotting points.
    value=replace_once(value,'assert [int(r["bin"]) for r in all_bins] == list(range(10)), (model, seed, calibration)',
                       'assert len({int(r["bin"]) for r in all_bins}) == len(all_bins) and all(0 <= int(r["bin"]) < 10 for r in all_bins), (model, seed, calibration)')
    value=replace_once(value,'for j, row in enumerate(all_bins):','for row in all_bins:\n                    j = int(row["bin"])')
    start=value.index('    for rel, digest in source_hashes.items():')
    end=value.index('    audit["source_sha256"]',start)
    value=value[:start]+'''    source_hashes = {'data/binary_curve_coordinates.csv': sha256(ROOT/'data/binary_curve_coordinates.csv')}
'''+value[end:]
    exec(compile(value,'<aggregate binary replay>','exec'),m.__dict__);m.main()
    # The archived plotting-audit prose describes its historical private-file
    # validation; replace it with the precise validation performed by this adapter.
    (out/'PLOTTING_AUDIT.md').write_text('Aggregate replay uses the packaged bin and rank coordinates. No predictions, model fitting, calibration fitting or hypothesis tests are recomputed. Numeric strings are round-trip checked.\n','utf-8')
    return {'figures':8,'data':'binary_curve_coordinates.csv'}

def methods(out):
    m=module('methods_flowchart');m.OUT=out
    m.build();m.validate();m.svg_export();m.drawio_export();m.pdf_export()
    subprocess.run([str(m.POPPLER),'-singlefile','-r','600','-png',str(out/f'{m.BASE}.pdf'),str(out/m.BASE)],check=True)
    return {'figures':1,'data':'self-contained geometric scene'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out','--output',dest='out',type=Path,required=True);p.add_argument('--edition',type=int,choices=[3000,5000],default=3000)
    p.add_argument('--font-regular',type=Path,required=True);p.add_argument('--font-bold',type=Path,required=True)
    p.add_argument('--node',default='node');p.add_argument('--sharp',default='sharp');p.add_argument('--pdftoppm',default='pdftoppm')
    p.add_argument('--groups',nargs='+',choices=['classification','segmentation','legacy','binary','methods'],default=['classification','segmentation','legacy','binary','methods'])
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for key,path in [('PUBLICATION_ARIAL_REGULAR',a.font_regular),('PUBLICATION_ARIAL_BOLD',a.font_bold)]:
        if not path.is_file():p.error('Locally licensed font file not found: '+str(path))
        os.environ[key]=str(path.resolve())
    os.environ.update(PUBLICATION_NODE=a.node,PUBLICATION_SHARP=a.sharp,PUBLICATION_PDFTOPPM=a.pdftoppm)
    results={}
    for group in a.groups:
        target=out/group;target.mkdir(exist_ok=True);results[group]=globals()[group](target)
    manifest=json.loads((HERE/f'provenance/figures_{a.edition}.json').read_text('utf-8'))
    by_name={p.name:p for p in out.rglob('*.png')}
    public=[r for r in manifest if not r['requires_private_images']]
    inventory=[]
    for r in public:
        name=r['original_file'];match=by_name.get(name)
        if match is None:continue
        digest=hashlib.sha256(match.read_bytes()).hexdigest()
        inventory.append({**r,'replayed_file':str(match.relative_to(out)).replace('\\','/'),'replayed_sha256':digest,'byte_identical_to_executed':digest==r['sha256']})
    if len(a.groups)==5:assert len(inventory)==62,(len(inventory),[r['original_file'] for r in public if r['original_file'] not in by_name])
    audit={'edition':a.edition,'groups':results,'replayed_public_figures':len(inventory),'private_qualitative_figures':6,'figure_inventory':inventory,'no_patient_rows_or_images_read':True,'no_model_execution':True}
    (out/'replay_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'replayed_public_figures':len(inventory),'groups':results,'byte_identical':sum(r['byte_identical_to_executed'] for r in inventory)}))

if __name__=='__main__':main()

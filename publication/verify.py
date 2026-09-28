"""Verify packaged aggregates, source hashes, table scope and optional figure replay."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math
import statistics

HERE=Path(__file__).resolve().parent

def read_json(path):
    return json.loads(path.read_text('utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(name):
    with (HERE/'data'/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-output',type=Path)
    a=p.parse_args()
    sources=read_json(HERE/'provenance/generator_sources.json')
    for record in sources:
        path=HERE/record['packaged_source']
        assert digest(path)==record['packaged_sha256'],path.name
        ast.parse(path.read_text('utf-8'))
    aggregates=read_json(HERE/'provenance/aggregate_sources.json')
    prohibited={'patient_id','case_id','frame_id','eye_id','unit_index','image_path','mask_path','audit_path'}
    for record in aggregates:
        assert digest(HERE/'data'/record['file'])==record['sha256'],record['file']
        data=rows(record['file'])
        assert len(data)==record['rows']
        assert not prohibited.intersection(record['columns'])
    editions={}
    for edition in (3000,5000):
        tables=read_json(HERE/f'tables/edition_{edition}.json')
        routed=read_json(HERE/f'tables/local_table_{edition}.json')
        assert len(tables)==30
        assert sum(r['section']=='main' for r in tables)==5
        assert routed['number'] not in [r['number'] for r in tables]
        assert len({r['number'] for r in tables})==30
        figures=read_json(HERE/f'provenance/figures_{edition}.json')
        public=[r for r in figures if not r['requires_private_images']]
        assert len(figures)==68 and len(public)==62
        identical=0
        if a.replay_output:
            outputs={f.name:f for f in a.replay_output.rglob('*.png')}
            for record in public:
                assert record['original_file'] in outputs,record['original_file']
                assert digest(outputs[record['original_file']])==record['sha256'],record['original_file']
                identical+=1
        editions[edition]={'public_tables':30,'local_table_routes':1,'public_figures':62,'private_figure_routes':6,'byte_identical_figures_checked':identical}
    # Independently reconstruct all 72 mean/sample-SD confusion cells from
    # integer counts, without loading participant-level predictions.
    cells=rows('confusion_cells_by_seed.csv');means=rows('confusion_mean_sd.csv')
    for record in means:
        names=['Normal','Abnormal'] if record['task']=='binary' else ['Normal','Papilledema','Pseudopapilledema']
        true=names[int(record['true_label'])]
        predicted=(names+['Abstain'])[int(record['predicted_label'])]
        group=[r for r in cells if r['task']==record['task'] and r['model']==record['model'] and r['true_class']==true and r['predicted_outcome']==predicted]
        assert len(group)==5
        values=[100*int(r['count'])/int(r['true_class_denominator']) for r in group]
        assert math.isclose(statistics.mean(values),float(record['mean_percentage']),abs_tol=1e-10)
        assert math.isclose(statistics.stdev(values),float(record['sample_sd_percentage_points']),abs_tol=1e-10)
    comparisons=rows('segmentation_all_360_paired_comparisons.csv')
    assert len(comparisons)==360
    families={r['family_id'] for r in comparisons}
    assert len(families)==12
    assert all(sum(r['family_id']==family for r in comparisons)==30 for family in families)
    print(json.dumps({'status':'passed','archived_generator_units':len(sources),'aggregate_csv_files':len(aggregates),'confusion_cells_verified':len(means),'segmentation_comparisons':len(comparisons),'editions':editions}))

if __name__=='__main__':
    main()

"""Eye CM-only revision using baseline plotting geometry and saved decisions.

No fitting, inference, selection, calibration, threshold changes, or writes to
experiment folders occur. Binary plots were already eye-level and are copied.
Only the MCC confusion plots are regenerated with the baseline plotting code.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import os
import shutil
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'publication_revision_20260926/classification'
V3 = ROOT / 'publication_revision_20260926_v3/classification'
sys.dont_write_bytecode = True

# Reuse the previously audited raw-eye validation, not its enlarged graphics.
spec = importlib.util.spec_from_file_location('eye_data_validator', Path(__file__).with_name('classification_eye_validator.py'))
eye = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eye)
eye.OUT = OUT
source = eye.source
source.OUT = OUT

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

FONT_RECORDS = []


def caption(task, main, model=None, seeds=None, part=None):
    original = json.loads((BASE / 'figure_index.json').read_text(encoding='utf-8'))
    slug = f'main_{task}_confusion_mean_sd' if main else f'supp_{task}_confusion_{model}_{part}'
    text = next(item['caption'] for item in original if item['id'] == slug)
    if task == 'binary':
        return text
    text = text.replace('patient-level', 'eye-level').replace('intended patient', 'intended eye')
    text = text.replace('threeclass', 'three-class')
    if main:
        text = text.replace('The diagonal reports failure-aware class recall (binary: specificity and sensitivity), agreeing with Tables 3 and 5 for the corresponding task.',
                            'Diagonals report failure-aware eye-level class recall; aggregate eye-level outcomes are in Supplementary Table S2. Each seed includes 20 normal, 8 papilledema, and 8 pseudopapilledema eyes.')
        text = text.replace('These are descriptive summaries of overlapping holdouts, not pooled independent observations;',
                            'Eyes within patients and overlapping seed holdouts are correlated; these summaries are descriptive, not pooled independent observations;')
    else:
        text = text.replace('True-class row totals are displayed in each panel heading and recorded in confusion_denominators.csv.',
                            'True-class rows contain 20 normal, 8 papilledema, and 8 pseudopapilledema eyes per seed.')
    return text


def svg_font_sizes(path):
    root = ET.parse(path).getroot()
    sizes = []
    for node in root.iter():
        if node.tag.rsplit('}', 1)[-1] != 'text':
            continue
        style = node.get('style', '')
        found = re.search(r'font-size:\s*([\d.]+)px', style)
        if not found:
            found = re.search(r'font:\s*(?:[^;]*?\s)?([\d.]+)px', style)
        if found:
            sizes.append(float(found.group(1)))
    return sizes


original_save = source.save


def save_original_style(fig, slug, category, task, passed_caption, model=None, class_name=None, **extra):
    """Validate native font sizes and geometry, without imposing new styling."""
    fig.canvas.draw()
    width, height = map(float, fig.get_size_inches())
    expected = (7.1, 7.4) if category == 'main_confusion' else (7.1, 9.8)
    source.check(f'baseline_physical_dimensions_{slug}', np.allclose([width, height], expected))
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in visible:
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            outside.append(artist.get_text())
    source.check(f'no_clipped_text_{slug}', not outside, outside)
    FONT_RECORDS.append({'id': slug, 'width_inches': width, 'height_inches': height,
                         'font_sizes_pt': sorted(set(float(t.get_fontsize()) for t in visible)),
                         'out_of_canvas_text': outside})
    original_save(fig, slug, category, task,
                  caption(task, category == 'main_confusion', model, extra.get('seeds'), extra.get('part')),
                  model=model, class_name=class_name, **extra)
    for ext in ['png', 'svg']:
        p = OUT / f'{slug}.{ext}'
        b = BASE / p.name
        if ext == 'png':
            source.check(f'baseline_pixel_dimensions_{slug}', Image.open(p).size == Image.open(b).size)
        else:
            source.check(f'baseline_all_text_font_sizes_{slug}', svg_font_sizes(p) == svg_font_sizes(b))


def main():
    baseline_paths = list(BASE.glob('*.png')) + list(BASE.glob('*.svg')) + [BASE / 'build_classification_figures.py', BASE / 'figure_index.json']
    baseline_hashes = {str(p): source.digest(p) for p in baseline_paths}
    bmetrics, pmetrics, classwise, tables, emetrics, patient_counts = eye.initialize()
    source.tracked(Path(__file__).with_name('classification_eye_validator.py'))

    # Restore precisely the baseline rcParams; the data-validator import set v3 values.
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    source.save = save_original_style

    # Execute the original function with only the requested analysis-unit changes.
    body = inspect.getsource(source.confusion_figures)
    body = body.replace("for task in ['binary', 'threeclass']:", "for task in ['threeclass']:", 1)
    body = body.replace("level = 'eye' if task == 'binary' else 'patient'", "level = 'eye'", 1)
    body = body.replace('Direct three-class classification: patient-level confusion',
                        'Direct three-class classification: eye-level confusion')
    exec(compile(body, '<baseline_CM_eye_revision>', 'exec'), source.__dict__)
    source.confusion_figures()

    # Binary confusion plots were eye-level already: retain exact bytes and appearance.
    original_index = json.loads((BASE / 'figure_index.json').read_text(encoding='utf-8'))
    for item in original_index:
        if item['task'] != 'binary' or item['category'] not in ['main_confusion', 'confusion_seed']:
            continue
        record = copy.deepcopy(item)
        for ext in ['png', 'svg']:
            src = BASE / f"{item['id']}.{ext}"
            dst = OUT / src.name
            shutil.copy2(src, dst)
            source.check(f'binary_bytes_identical_{src.name}', source.digest(src) == source.digest(dst))
            record[ext] = str(dst)
            record['files'][ext] = {'path': str(dst), 'sha256': source.digest(dst), 'size_bytes': dst.stat().st_size}
        record['caption'] = caption('binary', item['category'] == 'main_confusion', item.get('model'), item.get('seeds'), item.get('part'))
        record['level'] = 'eye'
        record['preserved_byte_identical_from_baseline'] = True
        source.FIGURES.append(record)

    # Independent derivative tables from the validated integer matrices.
    denoms, cells, summary = [], [], []
    for task in ['binary', 'threeclass']:
        names = ['Normal', 'Abnormal'] if task == 'binary' else source.CLASS_NAMES
        for model in source.MODELS:
            percentages = []
            for seed in source.SEEDS:
                matrix = source.COUNTS[task, model, seed]
                expected_n = [20, 16] if task == 'binary' else [20, 8, 8]
                source.check(f'intended_eye_rows_{task}_{model}_{seed}', np.array_equal(matrix.sum(axis=1), expected_n))
                perc = matrix / matrix.sum(axis=1, keepdims=True) * 100
                percentages.append(perc)
                for r, true in enumerate(names):
                    denoms.append({'task': task, 'level': 'eye', 'model': model, 'seed': seed, 'true_class': true,
                                   'n_intended_class': int(matrix[r].sum()), 'n_evaluable_class': int(matrix[r, :-1].sum()),
                                   'n_abstained_class': int(matrix[r, -1]), 'n_intended_all': int(matrix.sum())})
                    for c, pred in enumerate(names + ['Abstain']):
                        cells.append({'task': task, 'level': 'eye', 'model': model, 'seed': seed,
                                      'true_class': true, 'predicted_outcome': pred, 'count': int(matrix[r, c]),
                                      'true_class_denominator': int(matrix[r].sum()), 'row_percentage': float(perc[r, c])})
            arr = np.stack(percentages)
            for (r, c), mean in np.ndenumerate(arr.mean(axis=0)):
                summary.append({'task': task, 'level': 'eye', 'model': model, 'true_label': r, 'predicted_label': c,
                                'mean_percentage': float(mean), 'sample_sd_percentage_points': float(arr[:, r, c].std(ddof=1)),
                                'n_seeds': 5, 'ddof': 1})
    for name, rows in [('confusion_denominators', denoms), ('confusion_cells_by_seed', cells), ('confusion_mean_sd', summary)]:
        frame = pd.DataFrame(rows)
        reference = source.read_csv(V3 / f'{name}.csv')
        keys = [c for c in ['task', 'model', 'seed', 'true_class', 'predicted_outcome', 'true_label', 'predicted_label'] if c in frame.columns]
        frame = frame.sort_values(keys).reset_index(drop=True)
        reference = reference.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[reference.columns], reference, check_dtype=False, atol=1e-12, rtol=1e-12)
        source.check(f'exactly_matches_audited_v3_{name}', True)
        frame.to_csv(OUT / f'{name}.csv', index=False)

    for path, before in baseline_hashes.items():
        source.check(f'baseline_untouched_{Path(path).name}', source.digest(Path(path)) == before)
    for path, rec in source.SOURCES.items():
        source.check(f'raw_source_unchanged_{path}', source.digest(Path(rec['path'])) == rec['sha256'])
    source.FIGURES.sort(key=lambda r: (0 if r['category'] == 'main_confusion' else 1, r['task'], r.get('model') or '', r.get('part') or ''))
    (OUT / 'figure_index.json').write_text(json.dumps(source.FIGURES, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'font_geometry_audit.json').write_text(json.dumps(FONT_RECORDS, indent=2), encoding='utf-8')
    (OUT / 'consistency_tests.json').write_text(json.dumps({'passed': len(source.TESTS), 'tests': source.TESTS}, indent=2), encoding='utf-8')
    audit = {'baseline': str(BASE), 'scope': 'CM only. Binary copied byte-identically; three-class saved eye outcomes, unchanged plotting fonts and geometry.',
             'not_changed': 'Non-CM plots, baseline files, patient primary endpoint, experiments, thresholds, calibration, selection, and inference.',
             'figures': source.FIGURES, 'sources': source.SOURCES, 'font_geometry_audit': FONT_RECORDS,
             'passed_checks': len(source.TESTS), 'baseline_hashes': baseline_hashes}
    (OUT / 'audit_manifest.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'figures': len(source.FIGURES), 'passed_checks': len(source.TESTS), 'source_files': len(source.SOURCES)}))


if __name__ == '__main__':
    main()

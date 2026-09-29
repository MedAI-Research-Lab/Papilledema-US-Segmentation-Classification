"""Read-only reconstruction of publication figures from locked saved results."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BROOT = ROOT / 'strict_roi_results_4model_v1_2_0'
TROOT = ROOT / 'threeclass_roi_results_4model_v1_0_0'
MODELS = ['yolo26', 'vit_method2', 'emcad', 'sam2_unet']
DISPLAY = dict(zip(MODELS, ['YOLO26', 'ViT-Method2', 'EMCAD', 'SAM2-UNet']))
SEEDS = [17, 42, 2026, 3407, 9103]
COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
STYLES = ['-', '--', '-.', ':', (0, (5, 2, 1, 2))]
MARKERS = ['o', 's', '^', 'D', 'P']
CLASS_NAMES = ['Normal', 'Papilledema', 'Pseudopapilledema']
SHORT_NAMES = ['Normal', 'Papilledema', 'Pseudo-\npapilledema']
PROBCOLS = ['p_normal_calibrated', 'p_papilledema_calibrated', 'p_pseudopapilledema_calibrated']
SOURCES, FIGURES, TESTS = {}, [], []
COUNTS, PREDICTIONS, DENOMS, CELL_ROWS, CURVES, CURVE_STATS, RELIABILITY = {}, {}, [], [], [], [], []
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked(path):
    path = Path(path)
    key = str(path.relative_to(ROOT)).replace('\\', '/')
    SOURCES[key] = {'path': str(path), 'sha256': digest(path), 'size_bytes': path.stat().st_size}
    return path


def read_csv(path):
    path = tracked(path)
    table = pd.read_csv(path)
    SOURCES[str(path.relative_to(ROOT)).replace('\\', '/')]['rows'] = len(table)
    return table


def check(name, condition, detail=None):
    if not bool(condition):
        raise AssertionError(f'{name}: {detail}')
    TESTS.append({'name': name, 'status': 'passed', 'detail': detail})


def strict_bool(series):
    result = series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if result.isna().any():
        raise ValueError('Invalid boolean')
    return result.astype(bool)


def save(fig, slug, category, task, caption, model=None, class_name=None, **extra):
    files = {}
    for ext in ['png', 'svg']:
        path = OUT / f'{slug}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches=None)
        files[ext] = {'path': str(path), 'sha256': digest(path), 'size_bytes': path.stat().st_size}
    FIGURES.append({'id': slug, 'category': category, 'task': task, 'model': model,
                    'class_name': class_name, 'files': files, 'png': files['png']['path'],
                    'svg': files['svg']['path'], 'caption': caption, **extra})
    plt.close(fig)


def load_sources():
    binary_metrics = read_csv(BROOT / 'summary/classification_primary_model_specific_per_seed.csv')
    binary_metrics = binary_metrics.query("level == 'eye' and scope == 'ALL' and arm == 'predicted_roi'")
    check('binary_metric_grid_20', len(binary_metrics) == 20)
    binary_summary = read_csv(BROOT / 'summary/classification_primary_model_specific_five_seed_mean_sd.csv')
    binary_summary = binary_summary.query("level == 'eye' and scope == 'ALL' and arm == 'predicted_roi'")
    for model in MODELS:
        for seed in SEEDS:
            folder = BROOT / f'runs/{model}/seed_{seed}/evaluation/model_specific'
            cm = read_csv(folder / 'confusion_2x3.csv').query("level == 'eye' and scope == 'ALL'")
            matrix = cm.pivot(index='true_label', columns='outcome', values='n').reindex(index=[0, 1], columns=['negative', 'positive', 'abstain']).to_numpy(dtype=int)
            proportions = cm.pivot(index='true_label', columns='outcome', values='row_proportion').reindex(index=[0, 1], columns=['negative', 'positive', 'abstain']).to_numpy(float)
            pred = read_csv(folder / 'eyes.csv')
            pred['evaluable'] = strict_bool(pred.evaluable)
            check(f'binary_probability_scale_{model}_{seed}', set(pred.probability_scale) == {'temperature_scaled'})
            observed = np.zeros((2, 3), int)
            for row in pred.itertuples():
                observed[int(row.label), int(row.prediction) if row.evaluable else 2] += 1
            check(f'binary_saved_cm_matches_predictions_{model}_{seed}', np.array_equal(matrix, observed))
            check(f'binary_saved_row_proportion_{model}_{seed}', np.allclose(matrix / matrix.sum(axis=1, keepdims=True), proportions))
            metric = binary_metrics.query('model == @model and seed == @seed').iloc[0]
            diagonal = np.diag(matrix[:, :2]) / matrix.sum(axis=1)
            check(f'binary_table3_diagonal_recalls_{model}_{seed}', np.allclose(diagonal, [metric.failure_aware_specificity, metric.failure_aware_sensitivity]))
            check(f'binary_n_and_abstention_{model}_{seed}', len(pred) == metric.n_total and matrix[:, 2].sum() == metric.n_abstain)
            COUNTS['binary', model, seed] = matrix
            PREDICTIONS['binary', model, seed] = pred
        arr = np.stack([COUNTS['binary', model, s] / COUNTS['binary', model, s].sum(axis=1, keepdims=True) for s in SEEDS])
        summary = binary_summary.query('model == @model').iloc[0]
        for k, field in [(0, 'failure_inclusive.specificity'), (1, 'failure_inclusive.sensitivity')]:
            check(f'binary_summary_{model}_{field}', np.allclose([arr[:, k, k].mean(), arr[:, k, k].std(ddof=1)], [summary[f'{field}_mean'], summary[f'{field}_sd']]))

    manifest = json.loads(tracked(TROOT / 'summary/publication_output_manifest.json').read_text(encoding='utf-8'))
    tables = {}
    for name in ['patient_predictions.csv', 'patient_confusion_3x4.csv', 'patient_metrics.csv', 'classwise_metrics.csv', 'risk_coverage.csv', 'calibration_metrics.csv']:
        path = TROOT / 'tables' / name
        data = read_csv(path)
        record = manifest['tables'][name]
        check(f'threeclass_manifest_hash_{name}', digest(path) == record['sha256'] and len(data) == record['rows'])
        data = data.loc[data.classifier_strategy == 'model_specific'].copy()
        tables[name] = data
    predictions = tables['patient_predictions.csv']
    predictions['abstained'] = strict_bool(predictions.abstained)
    three_metrics = tables['patient_metrics.csv'].query("probability_scale == 'calibrated'")
    three_classwise = tables['classwise_metrics.csv']
    for model in MODELS:
        for seed in SEEDS:
            cm = tables['patient_confusion_3x4.csv'].query('model == @model and seed == @seed')
            matrix = cm.pivot(index='true_label', columns='predicted_label', values='count').reindex(index=range(3), columns=range(4)).to_numpy(dtype=int)
            pred = predictions.query('model == @model and seed == @seed').copy()
            observed = np.zeros((3, 4), int)
            for row in pred.itertuples():
                observed[int(row.true_label), int(row.predicted_label)] += 1
            check(f'threeclass_saved_cm_matches_predictions_{model}_{seed}', np.array_equal(matrix, observed))
            check(f'threeclass_abstentions_{model}_{seed}', np.array_equal(pred.abstained, pred.predicted_label == 3))
            probs = pred.loc[~pred.abstained, PROBCOLS].to_numpy(float)
            check(f'threeclass_probabilities_{model}_{seed}', np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1) and np.array_equal(probs.argmax(axis=1), pred.loc[~pred.abstained, 'predicted_label']))
            classwise = three_classwise.query('model == @model and seed == @seed').sort_values('class_label')
            diagonal = np.diag(matrix[:, :3]) / matrix.sum(axis=1)
            check(f'threeclass_table5_diagonal_recalls_{model}_{seed}', np.allclose(diagonal, classwise.failure_aware_recall))
            metric = three_metrics.query('model == @model and seed == @seed').iloc[0]
            check(f'threeclass_BA_{model}_{seed}', np.isclose(diagonal.mean(), metric.failure_aware_balanced_accuracy))
            check(f'threeclass_n_and_abstention_{model}_{seed}', len(pred) == metric.n_intended and matrix[:, 3].sum() == metric.n_abstained)
            COUNTS['threeclass', model, seed] = matrix
            PREDICTIONS['threeclass', model, seed] = pred
    return binary_metrics, three_metrics, three_classwise, tables


def cm_axis(ax, matrix, task, sd=None, counts=None, fontsize=12):
    labels = ['Normal', 'Abnormal'] if task == 'binary' else SHORT_NAMES
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(labels) + 1), labels + ['Abstain'])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Predicted outcome', labelpad=8)
    ax.set_ylabel('True class', labelpad=10)
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.3)
    ax.tick_params(which='both', length=0)
    for (r, c), value in np.ndenumerate(matrix):
        label = f'{value:.1f}%\n± {sd[r,c]:.1f}' if sd is not None else f'{value:.1f}%\n(n = {counts[r,c]})'
        ax.text(c, r, label, ha='center', va='center', fontsize=fontsize,
                color='white' if value >= 58 else '#182B3A', linespacing=1.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def confusion_figures():
    for task in ['binary', 'threeclass']:
        level = 'eye' if task == 'binary' else 'patient'
        classes = ['Normal', 'Abnormal'] if task == 'binary' else CLASS_NAMES
        means, sds = {}, {}
        for model in MODELS:
            percentages = []
            for seed in SEEDS:
                matrix = COUNTS[task, model, seed]
                totals = matrix.sum(axis=1)
                perc = 100 * matrix / totals[:, None]
                percentages.append(perc)
                for r, label in enumerate(classes):
                    DENOMS.append({'task': task, 'level': level, 'model': model, 'seed': seed, 'true_class': label,
                                   'n_intended_class': int(totals[r]), 'n_abstained_class': int(matrix[r,-1]), 'n_evaluable_class': int(totals[r] - matrix[r,-1]), 'n_intended_all': int(totals.sum())})
                    for c, outcome in enumerate(classes + ['Abstain']):
                        CELL_ROWS.append({'task': task, 'level': level, 'model': model, 'seed': seed, 'true_class': label, 'predicted_outcome': outcome, 'count': int(matrix[r,c]), 'true_class_denominator': int(totals[r]), 'row_percentage': float(perc[r,c])})
            array = np.stack(percentages)
            means[model], sds[model] = array.mean(axis=0), array.std(axis=0, ddof=1)
            check(f'{task}_{model}_mean_row_sums_100', np.allclose(means[model].sum(axis=1), 100))
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.4))
        fig.subplots_adjust(left=.145, right=.94, top=.815, bottom=.25, hspace=.70, wspace=.78)
        for i, (ax, model) in enumerate(zip(axes.flat, MODELS)):
            im = cm_axis(ax, means[model], task, sd=sds[model], fontsize=9)
            ax.tick_params(axis='both',labelsize=8)
            ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
            ax.set_ylabel('')
            if task == 'threeclass':
                ax.set_xticklabels(['Normal', 'PE', 'PPE', 'Abstain'])
            ax.set_title(f'{chr(65+i)}  {DISPLAY[model]}', loc='left', fontweight='bold', fontsize=10.5, pad=11)
        cb = fig.colorbar(im, cax=fig.add_axes([.22, .125, .61, .016]), orientation='horizontal')
        cb.set_label('Within-seed true-row percentage (%)', labelpad=5,fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.text(.025, .54, 'True class', rotation=90, va='center', fontsize=9)
        title = 'Binary classification: eye-level confusion' if task == 'binary' else 'Direct three-class classification: patient-level confusion'
        fig.suptitle(title.replace(': ', ':\n'), y=.975, fontsize=13, fontweight='bold',linespacing=1.3)
        fig.text(.5, .885, 'Primary model-specific strategy | five held-out splits', ha='center', fontsize=9)
        footer = 'Each cell: mean ± sample SD across 5 seeds (percentage points).\nAbstentions included; overlapping seed holdouts are descriptive.'
        if task == 'threeclass':
            footer += '\nPE, papilledema; PPE, pseudopapilledema.'
        fig.text(.5, .012, footer, ha='center', fontsize=8, linespacing=1.4)
        caption = f'{title}. Each matrix uses the primary model-specific classifier. Cells show the arithmetic mean ± sample SD (ddof = 1) of within-seed true-row percentages for seeds 17, 42, 2026, 3407, and 9103. True rows include every intended {level}, including abstentions, and sum to 100% before rounding. The colour scale is fixed at 0–100% for every panel. The diagonal reports failure-aware class recall (binary: specificity and sensitivity), agreeing with Tables 3 and 5 for the corresponding task. These are descriptive summaries of overlapping holdouts, not pooled independent observations; SD is in percentage points.'
        save(fig, f'main_{task}_confusion_mean_sd', 'main_confusion', task, caption, means={m: means[m].tolist() for m in MODELS}, sample_sds={m: sds[m].tolist() for m in MODELS}, class_order=classes, outcome_order=classes+['Abstain'], seeds=SEEDS, level=level, sd_ddof=1, colour_limits=[0,100])
        for model in MODELS:
            for page, seeds in [('A', SEEDS[:3]), ('B', SEEDS[3:])]:
                fig, axes = plt.subplots(3, 1, figsize=(7.1, 9.8))
                fig.subplots_adjust(left=.23, right=.82, top=.86, bottom=.14, hspace=.85)
                for ax, seed in zip(axes, seeds):
                    matrix = COUNTS[task, model, seed]
                    im = cm_axis(ax, 100*matrix/matrix.sum(axis=1, keepdims=True), task, counts=matrix, fontsize=12)
                    totals = ', '.join(str(n) for n in matrix.sum(axis=1))
                    ax.set_title(f'Seed {seed} | true-row n: {totals}', loc='left', fontsize=11, fontweight='bold', pad=12)
                if len(seeds) == 2:
                    axes[2].axis('off')
                cb = fig.colorbar(im, cax=fig.add_axes([.86, .28, .018, .45]))
                cb.set_label('Within-seed true-row percentage (%)', labelpad=8,fontsize=9.5)
                cb.ax.tick_params(labelsize=9)
                fig.suptitle(f'{DISPLAY[model]} | {"Binary" if task == "binary" else "Direct three-class"} confusion', fontsize=15, fontweight='bold', y=.96)
                fig.text(.5, .92, f'Primary model-specific strategy | {level} level | page {page}', ha='center', fontsize=11)
                fig.text(.5, .025, 'Cells show true-row % and saved counts; abstention remains an outcome.\nRows include all intended observations. Five overlapping seed holdouts are not pooled.', ha='center', fontsize=9.5, linespacing=1.4)
                save(fig, f'supp_{task}_confusion_{model}_{page}', 'confusion_seed', task,
                     f'{DISPLAY[model]}: {task} {level}-level confusion, page {page}. Seed panels {seeds} are arranged in three rows and one column; page B leaves the third slot empty. Each cell shows the saved count and percentage of the complete true-class row, including abstentions. A common 0–100% scale applies to all matrices. True-class row totals are displayed in each panel heading and recorded in confusion_denominators.csv. Primary model-specific strategy only; no pooling across seeds.', model=model, seeds=seeds, part=page, level=level, layout='3 rows x 1 column', empty_slots=3-len(seeds))


def unit_axes(ax, xlabel, ylabel):
    ax.set(xlim=(0,1), ylim=(0,1.02), xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
    ax.grid(True, color='#DDE3E8', lw=.7)
    ax.spines[['top','right']].set_visible(False)


def legend(fig, y=.05, markers=False):
    handles = [Line2D([0],[0], color=COLOURS[i], ls=STYLES[i], marker=MARKERS[i] if markers else None, lw=2, label=f'Seed {s}') for i,s in enumerate(SEEDS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,y), ncol=3, frameon=False, fontsize=10)


def discrimination(binary_metrics, three_classwise):
    for task in ['binary', 'threeclass']:
        level = 'eye' if task == 'binary' else 'patient'
        for model in MODELS:
            for class_index in ([1] if task == 'binary' else range(3)):
                cname = 'Abnormal' if task == 'binary' else CLASS_NAMES[class_index]
                fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.4))
                fig.subplots_adjust(left=.14, right=.95, top=.83, bottom=.18, hspace=.46)
                for i, seed in enumerate(SEEDS):
                    pred = PREDICTIONS[task, model, seed]
                    valid = pred.loc[pred.evaluable] if task == 'binary' else pred.loc[~pred.abstained]
                    truth = valid.label.to_numpy(int) if task == 'binary' else (valid.true_label == class_index).to_numpy(int)
                    prob = valid.probability.to_numpy(float) if task == 'binary' else valid[PROBCOLS[class_index]].to_numpy(float)
                    fpr, tpr, _ = roc_curve(truth, prob)
                    precision, recall, _ = precision_recall_curve(truth, prob)
                    auroc, ap = roc_auc_score(truth, prob), average_precision_score(truth, prob)
                    axes[0].plot(fpr, tpr, color=COLOURS[i], ls=STYLES[i], lw=1.8, label=f'{seed}: {auroc:.3f}')
                    axes[1].plot(recall, precision, color=COLOURS[i], ls=STYLES[i], lw=1.8, label=f'{seed}: {ap:.3f}')
                    CURVE_STATS.append({'task':task,'model':model,'seed':seed,'class':cname,'n_intended':len(pred),'n_evaluable':len(valid),'n_positive':int(truth.sum()),'n_negative':int(len(truth)-truth.sum()),'auroc':float(auroc),'average_precision':float(ap)})
                    for ctype, x, y in [('ROC',fpr,tpr),('PR',recall,precision)]:
                        CURVES.extend({'task':task,'model':model,'seed':seed,'class':cname,'curve':ctype,'point':j,'x':float(a),'y':float(b)} for j,(a,b) in enumerate(zip(x,y)))
                    if task == 'binary':
                        metric = binary_metrics.query('model == @model and seed == @seed').iloc[0]
                        ref = [metric['conditional.auroc'], metric['conditional.average_precision']]
                    else:
                        metric = three_classwise.query('model == @model and seed == @seed and class_label == @class_index').iloc[0]
                        ref = [metric.ovr_auroc, metric.ovr_average_precision]
                    check(f'{task}_{model}_{seed}_{cname}_discrimination_saved_metrics', np.allclose([auroc,ap], ref, atol=1e-9))
                axes[0].plot([0,1],[0,1], color='#707070', ls=':', lw=1)
                unit_axes(axes[0], 'False-positive rate (1 - specificity)', 'Sensitivity')
                unit_axes(axes[1], 'Recall (sensitivity)', 'Precision')
                axes[0].set_title('A  Receiver operating characteristic', loc='left', fontweight='bold', pad=10)
                axes[1].set_title('B  Precision–recall', loc='left', fontweight='bold', pad=10)
                axes[0].legend(title='Seed: AUROC', loc='lower right', framealpha=.95, fontsize=9, title_fontsize=10)
                axes[1].legend(title='Seed: average precision', loc='lower left', framealpha=.95, fontsize=9, title_fontsize=10)
                fig.suptitle(f'{DISPLAY[model]} | {cname}', y=.965, fontsize=15, fontweight='bold')
                fig.text(.5, .91, f'{"Binary" if task == "binary" else "Three-class one-vs-rest"} discrimination | {level} level\nCalibrated probabilities | primary model-specific strategy', ha='center', fontsize=11, linespacing=1.5)
                fig.text(.5, .055, 'Each trace represents one seed among evaluable observations.\nAbstentions are excluded from ROC/PR; overlapping holdouts are not pooled.', ha='center', fontsize=9.5, linespacing=1.5)
                save(fig, f'supp_{task}_roc_pr_{model}_class{class_index}', 'roc_pr', task,
                     f'{DISPLAY[model]} {cname} {"versus normal" if task == "binary" else "one-versus-rest"} discrimination at {level} level. ROC is above precision–recall, with five seed-specific traces derived solely from saved calibrated probabilities among evaluable observations. No new classifier, calibration, threshold, or model selection was fitted. Legend values are seed-specific AUROC and average precision; all denominators are in discrimination_statistics.csv. Structural abstentions have no probability and are excluded. Seed holdouts overlap and remain separate.', model=model, class_name=cname, seeds=SEEDS, level=level, layout='2 rows x 1 column')


def reliability_and_risk(tables):
    edges = np.linspace(0,1,6)
    risk = tables['risk_coverage.csv']
    for model in MODELS:
        fig, axes = plt.subplots(3,1,figsize=(7.2,11.6))
        fig.subplots_adjust(left=.17,right=.95,top=.89,bottom=.16,hspace=.55)
        for c, ax in enumerate(axes):
            for i, seed in enumerate(SEEDS):
                pred = PREDICTIONS['threeclass',model,seed]
                pred = pred.loc[~pred.abstained]
                p = pred[PROBCOLS[c]].to_numpy(float)
                y = (pred.true_label == c).to_numpy(int)
                bins = np.clip(np.searchsorted(edges,p,side='right')-1,0,4)
                xs, ys = [], []
                for b in range(5):
                    keep = bins == b
                    if keep.any():
                        xs.append(p[keep].mean()); ys.append(y[keep].mean())
                        RELIABILITY.append({'model':model,'seed':seed,'class':CLASS_NAMES[c],'bin':b,'bin_lower':edges[b],'bin_upper':edges[b+1],'n':int(keep.sum()),'mean_probability':float(xs[-1]),'observed_frequency':float(ys[-1])})
                ax.plot(xs,ys,color=COLOURS[i],ls=STYLES[i],marker=MARKERS[i],ms=4,lw=1.5)
            ax.plot([0,1],[0,1], color='#777777', ls=':',lw=1)
            unit_axes(ax,'Mean predicted probability','Observed class frequency')
            ax.set_title(f'{chr(65+c)}  {CLASS_NAMES[c]}', loc='left',fontweight='bold',pad=9)
        fig.suptitle(f'{DISPLAY[model]} | Patient-level reliability',y=.965,fontsize=15,fontweight='bold')
        fig.text(.5,.928,'Calibrated probabilities | primary model-specific strategy',ha='center',fontsize=11)
        legend(fig,y=.06,markers=True)
        fig.text(.5,.015,'Five fixed-width bins per seed; empty bins omitted.\nEvaluable patients only. Seed holdouts overlap and are not pooled.',ha='center',fontsize=9.5,linespacing=1.4)
        save(fig,f'supp_threeclass_reliability_{model}','reliability','threeclass',
             f'{DISPLAY[model]} classwise patient reliability, arranged as three rows and one column: normal, papilledema, and pseudopapilledema. Five fixed-width bins [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), and [0.8,1] match the previous saved-figure definition; empty bins are omitted. Each point pairs mean saved calibrated class probability with observed class frequency, conditional on evaluable patients. Five seeds remain separate. The diagonal indicates perfect calibration; exact per-bin counts and points are in reliability_points.csv.',model=model,seeds=SEEDS,layout='3 rows x 1 column',bin_edges=edges.tolist())

        fig, ax = plt.subplots(figsize=(7.2,6.0))
        fig.subplots_adjust(left=.14,right=.96,top=.79,bottom=.27)
        for i, seed in enumerate(SEEDS):
            group = risk.query('model == @model and seed == @seed').sort_values('rank')
            pred = PREDICTIONS['threeclass',model,seed]
            check(f'risk_coverage_denominator_{model}_{seed}', len(group)==len(pred) and np.allclose(group.coverage, np.arange(1,len(pred)+1)/len(pred)))
            endrisk = 1-np.diag(COUNTS['threeclass',model,seed][:,:3]).sum()/len(pred)
            check(f'risk_terminal_failure_aware_{model}_{seed}', np.isclose(group.selective_risk.iloc[-1],endrisk))
            ax.step(group.coverage, group.selective_risk,where='post',color=COLOURS[i],ls=STYLES[i],lw=1.8)
        unit_axes(ax,'Coverage (fraction of intended patients retained)','Selective error rate')
        fig.suptitle(f'{DISPLAY[model]} | Patient risk–coverage',y=.955,fontsize=15,fontweight='bold')
        fig.text(.5,.87,'Failure-aware | calibrated primary model-specific strategy',ha='center',fontsize=11)
        legend(fig,y=.105)
        fig.text(.5,.025,'Saved maximum-probability ranking; structural abstentions appended\nat lowest confidence and counted as errors. Seeds remain separate.',ha='center',fontsize=9.5,linespacing=1.4)
        save(fig,f'supp_threeclass_risk_coverage_{model}','risk_coverage','threeclass',
             f'{DISPLAY[model]} patient risk–coverage using the saved overall three-class curves. Patients are ranked within seed by maximum calibrated class probability. Structural abstentions are appended at lowest confidence and count as errors; coverage divides retained patients by all intended patients. The panel preserves the original overall-error definition rather than inventing class-specific curves. All five seed traces remain separate; terminal risk equals one minus failure-aware accuracy.',model=model,seeds=SEEDS,layout='1 panel',confidence_definition='maximum_calibrated_patient_probability')


def performance_supplement(binary_metrics, three_metrics):
    for task, table in [('binary',binary_metrics),('threeclass',three_metrics)]:
        fig,axes=plt.subplots(2,1,figsize=(7.8,8.5))
        fig.subplots_adjust(left=.13,right=.95,top=.85,bottom=.17,hspace=.48)
        for ax,field,title in zip(axes,['failure_aware_balanced_accuracy','coverage'],['Failure-aware balanced accuracy','Coverage']):
            for m,model in enumerate(MODELS):
                group=table.query('model == @model').set_index('seed').loc[SEEDS]
                values=100*group[field].to_numpy(float)
                for s,value in enumerate(values):
                    ax.scatter(m+(s-2)*.055,value,color=COLOURS[s],marker=MARKERS[s],s=34,zorder=3)
                ax.plot([m-.17,m+.17],[values.mean()]*2,color='#111111',lw=2.5)
                ax.plot([m,m],[values.min(),values.max()],color='#666666',lw=1.2,zorder=1)
            ax.set(ylim=(0,104),ylabel=f'{title} (%)')
            ax.set_xticks(range(4),[DISPLAY[m] for m in MODELS])
            ax.grid(axis='y',color='#DDE3E8',lw=.7)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'{"Binary eye-level" if task=="binary" else "Three-class patient-level"} performance',y=.96,fontsize=15,fontweight='bold')
        fig.text(.5,.90,'Primary model-specific strategy | five held-out splits',ha='center',fontsize=11)
        legend(fig,y=.057,markers=True)
        fig.text(.5,.013,'Points: seed results. Black bar: arithmetic mean. Vertical line: observed range.\nOverlapping seed holdouts are descriptive, not independent replications.',ha='center',fontsize=9.5,linespacing=1.4)
        save(fig,f'supp_{task}_failure_aware_ba_coverage','supplement_performance',task,
             f'Failure-aware balanced accuracy and coverage for {task} primary model-specific classification. Seed-specific points retain the five overlapping held-out splits; horizontal black bars show arithmetic means and vertical grey lines show observed minimum–maximum ranges, not confidence intervals. Binary results use eye-level ALL scope; direct three-class results use calibrated patient-level results. Abstentions count as failures in balanced accuracy.',seeds=SEEDS,layout='2 rows x 1 column')


def finish():
    for name, rows in [('confusion_denominators',DENOMS),('confusion_cells_by_seed',CELL_ROWS),('discrimination_statistics',CURVE_STATS),('discrimination_curve_points',CURVES),('reliability_points',RELIABILITY)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    summary=[]
    for task in ['binary','threeclass']:
        for model in MODELS:
            values=np.stack([COUNTS[task,model,s]/COUNTS[task,model,s].sum(axis=1,keepdims=True)*100 for s in SEEDS])
            for (r,c),mean in np.ndenumerate(values.mean(axis=0)):
                summary.append({'task':task,'model':model,'true_label':r,'predicted_label':c,'mean_percentage':float(mean),'sample_sd_percentage_points':float(values[:,r,c].std(ddof=1)),'n_seeds':5,'ddof':1})
    pd.DataFrame(summary).to_csv(OUT/'confusion_mean_sd.csv',index=False)
    # Ensure inputs still match initial hashes after all derived figure production.
    for name, record in SOURCES.items():
        check(f'source_unchanged_{name}',digest(Path(record['path']))==record['sha256'])
    payload={'created_for':'publication_revision_20260926','strategy':'model_specific','models':MODELS,'seeds':SEEDS,
             'source_policy':'Saved predictions, matrices, metrics and curves only. No fitting or model/threshold selection.',
             'aggregation':'Within-seed true-row percentages; across-seed arithmetic mean and sample SD ddof=1. No pooling.',
             'sources':SOURCES,'consistency_tests':TESTS,'figures':FIGURES}
    (OUT/'audit_manifest.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'figure_index.json').write_text(json.dumps(FIGURES,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'consistency_tests.json').write_text(json.dumps({'passed':len(TESTS),'tests':TESTS},indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(FIGURES),'outputs':2*len(FIGURES),'passed_checks':len(TESTS),'sources':len(SOURCES)}))


if __name__=='__main__':
    binary_metrics,three_metrics,three_classwise,tables=load_sources()
    confusion_figures()
    discrimination(binary_metrics,three_classwise)
    reliability_and_risk(tables)
    performance_supplement(binary_metrics,three_metrics)
    finish()

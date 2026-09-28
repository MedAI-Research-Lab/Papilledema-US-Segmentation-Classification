"""Publication revision v3: saved-result eye confusion and >=17 pt graphics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
old = Path(__file__).with_name('classification.py')
spec = importlib.util.spec_from_file_location('validated_saved_sources', old)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

MODELS, DISPLAY, SEEDS = source.MODELS, source.DISPLAY, source.SEEDS
COLOURS, STYLES, MARKERS = source.COLOURS, source.STYLES, source.MARKERS
NAMES, PROBCOLS = source.CLASS_NAMES, source.PROBCOLS
MIN_FONT = 17.0
FIGURES, FONT_AUDIT, DENOMS, CELL_ROWS, CURVE_STATS, CURVES, RELIABILITY = [], [], [], [], [], [], []
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.titlesize':17,
                     'axes.labelsize':17,'xtick.labelsize':17,'ytick.labelsize':17,
                     'legend.fontsize':17,'legend.title_fontsize':17,
                     'figure.titlesize':20,'svg.fonttype':'none','savefig.facecolor':'white'})
read_csv, check = source.read_csv, source.check
tracked = source.tracked


def save(fig, slug, category, task, caption, model=None, class_name=None, **extra):
    fig.canvas.draw()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    fonts = [float(t.get_fontsize()) for t in visible]
    check(f'font_minimum_{slug}', min(fonts)>=MIN_FONT)
    width,height=map(float,fig.get_size_inches())
    check(f'canvas_limit_{slug}',width<=7.00001 and height<=8.50001)
    renderer=fig.canvas.get_renderer()
    outside=[]
    for t in visible:
        bbox=t.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1:
            outside.append(t.get_text())
    check(f'no_clipped_text_{slug}',not outside, outside)
    record={'id':slug,'minimum_font_pt':min(fonts),'maximum_font_pt':max(fonts),'width_inches':width,
            'height_inches':height,'n_visible_text_artists':len(visible),'out_of_canvas_text':outside,
            'minimum_pdf_embedding_scale_for_16pt':16/min(fonts)}
    FONT_AUDIT.append(record)
    files={}
    for ext in ['png','svg']:
        p=OUT/f'{slug}.{ext}'
        fig.savefig(p,dpi=300)
        files[ext]={'path':str(p),'sha256':source.digest(p),'size_bytes':p.stat().st_size}
    FIGURES.append({'id':slug,'category':category,'task':task,'model':model,'class_name':class_name,
                    'png':files['png']['path'],'svg':files['svg']['path'],'files':files,'caption':caption,
                    'font_audit':record,**extra})
    plt.close(fig)


def initialize():
    source.OUT=OUT
    bmetrics,pmetrics,classwise,tables=source.load_sources()
    source.tracked(old)
    patient_counts={k:v.copy() for k,v in source.COUNTS.items() if k[0]=='threeclass'}
    eyes=read_csv(source.TROOT/'tables/classification_per_seed.csv')
    eyes=eyes.query("strategy == 'model_specific' and probability_state == 'calibrated' and level == 'eye'").copy()
    check('threeclass_calibrated_eye_metrics_20',len(eyes)==20)
    for model in MODELS:
        for seed in SEEDS:
            folder=source.TROOT/f'runs/{model}/seed_{seed}/evaluation/model_specific'
            pred=read_csv(folder/'eyes_calibrated.csv')
            pred['evaluable']=source.strict_bool(pred.evaluable)
            saved=read_csv(folder/'confusion_3x4_eye_calibrated.csv')
            matrix=saved.pivot(index='actual_class',columns='predicted_class',values='count').reindex(index=range(3),columns=range(4)).to_numpy(int)
            observed=np.zeros((3,4),int)
            for row in pred.itertuples():
                observed[int(row.label_3class),int(row.prediction) if row.evaluable else 3]+=1
            check(f'eye_threeclass_saved_counts_match_{model}_{seed}',np.array_equal(matrix,observed))
            check(f'eye_threeclass_intended_rows_{model}_{seed}',np.array_equal(matrix.sum(axis=1),[20,8,8]))
            probabilities=pred.loc[pred.evaluable,['probability_0','probability_1','probability_2']].to_numpy(float)
            check(f'eye_threeclass_argmax_saved_{model}_{seed}',np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1),1) and np.array_equal(probabilities.argmax(axis=1),pred.loc[pred.evaluable,'prediction']))
            props=saved.pivot(index='actual_class',columns='predicted_class',values='row_proportion').reindex(index=range(3),columns=range(4)).to_numpy(float)
            check(f'eye_threeclass_saved_row_percentages_{model}_{seed}',np.allclose(matrix/matrix.sum(axis=1,keepdims=True),props))
            metric=eyes.query('model == @model and seed == @seed').iloc[0]
            diag=np.diag(matrix[:,:3])/matrix.sum(axis=1)
            check(f'eye_threeclass_metrics_diagonal_{model}_{seed}',np.allclose(diag,[metric[f'failure_aware.per_class.{c}.recall'] for c in ['normal','papilledema','pseudopapilledema']]))
            check(f'eye_threeclass_metrics_n_BA_coverage_{model}_{seed}',len(pred)==metric.n_total and matrix[:,-1].sum()==metric.n_abstain and np.isclose(diag.mean(),metric.failure_aware_balanced_accuracy) and np.isclose(pred.evaluable.mean(),metric.coverage))
            source.COUNTS['threeclass',model,seed]=matrix
    return bmetrics,pmetrics,classwise,tables,eyes,patient_counts


def record_eye_metrics(binary,three):
    records=[]
    for task,table in [('binary',binary),('threeclass',three)]:
        for model in MODELS:
            for seed in SEEDS:
                cm=source.COUNTS[task,model,seed]
                g=table.query('model == @model and seed == @seed').iloc[0]
                d={'task':task,'level':'eye','model':model,'seed':seed,'strategy':'model_specific','probability_state':'calibrated','n_total':int(cm.sum()),'n_evaluable':int(cm[:,:-1].sum()),'n_abstained':int(cm[:,-1].sum()),'coverage':float(cm[:,:-1].sum()/cm.sum()),'failure_aware_accuracy':float(np.trace(cm[:,:-1])/cm.sum()),'failure_aware_balanced_accuracy':float(np.mean(np.diag(cm[:,:-1])/cm.sum(axis=1)))}
                names=['normal','abnormal'] if task=='binary' else ['normal','papilledema','pseudopapilledema']
                for c,name in enumerate(names):
                    d[f'{name}_n_intended']=int(cm[c].sum())
                    d[f'{name}_failure_aware_recall']=float(cm[c,c]/cm[c].sum())
                if task=='binary':
                    d['conditional_auroc']=float(g['conditional.auroc']);d['conditional_average_precision']=float(g['conditional.average_precision'])
                records.append(d)
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'eye_metrics_per_seed.csv',index=False)
    aggregate=[]
    excluded={'task','level','model','seed','strategy','probability_state'}
    for (task,model),group in frame.groupby(['task','model']):
        for metric in [c for c in frame.columns if c not in excluded]:
            values=group[metric].dropna().to_numpy(float)
            if len(values):
                aggregate.append({'task':task,'level':'eye','model':model,'metric':metric,'n_seeds':len(values),'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'ddof':1})
    pd.DataFrame(aggregate).to_csv(OUT/'eye_metrics_mean_sd.csv',index=False)
    three.to_csv(OUT/'threeclass_eye_metrics_saved_rows.csv',index=False)


def matrix_axis(ax,values,task,sd=None,counts=None):
    labels=['N','D'] if task=='binary' else ['N','P','PP']
    im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(labels)+1),labels+['A'])
    ax.set_yticks(range(len(labels)),labels)
    ax.set_xticks(np.arange(-.5,values.shape[1],1),minor=True)
    ax.set_yticks(np.arange(-.5,values.shape[0],1),minor=True)
    ax.tick_params(which='both',length=0,pad=5)
    ax.grid(which='minor',color='white',lw=1.5)
    for spine in ax.spines.values():spine.set_visible(False)
    annotations=[]
    for (r,c),value in np.ndenumerate(values):
        label=f'{value:.1f}\n±{sd[r,c]:.1f}' if sd is not None else f'{value:.1f}%\nn={counts[r,c]}'
        annotations.append(ax.text(c,r,label,ha='center',va='center',fontsize=17,fontstretch='condensed',linespacing=1.06,color='white' if value>=58 else '#182B3A'))
    return im,annotations


def confusion():
    for task in ['binary','threeclass']:
        classes=['Normal','Abnormal'] if task=='binary' else NAMES
        means,sds={},{}
        for model in MODELS:
            arr=[]
            for seed in SEEDS:
                counts=source.COUNTS[task,model,seed]
                percentages=100*counts/counts.sum(axis=1,keepdims=True);arr.append(percentages)
                for r,name in enumerate(classes):
                    DENOMS.append({'task':task,'level':'eye','model':model,'seed':seed,'true_class':name,'n_intended_class':int(counts[r].sum()),'n_evaluable_class':int(counts[r,:-1].sum()),'n_abstained_class':int(counts[r,-1]),'n_intended_all':int(counts.sum())})
                    for c,outcome in enumerate(classes+['Abstain']):
                        CELL_ROWS.append({'task':task,'level':'eye','model':model,'seed':seed,'true_class':name,'predicted_outcome':outcome,'count':int(counts[r,c]),'true_class_denominator':int(counts[r].sum()),'row_percentage':float(percentages[r,c])})
            means[model],sds[model]=np.mean(arr,axis=0),np.std(arr,axis=0,ddof=1)
            check(f'eye_{task}_mean_rows_100_{model}',np.allclose(means[model].sum(axis=1),100))
        fig=plt.figure(figsize=(7,8.5))
        positions=[(.07,.57,.435,.255),(.555,.57,.435,.255),(.07,.20,.435,.255),(.555,.20,.435,.255)]
        all_annotations=[]
        for i,(model,pos) in enumerate(zip(MODELS,positions)):
            ax=fig.add_axes(pos);im,anns=matrix_axis(ax,means[model],task,sd=sds[model]);all_annotations.extend(anns)
            ax.set_title(DISPLAY[model],loc='left',fontsize=17,fontweight='bold',pad=12)
        fig.suptitle(f'{"Binary" if task=="binary" else "Three-class"} eye-level confusion',y=.97,fontsize=20,fontweight='bold')
        fig.text(.5,.903,'Rows: true | columns: predicted',ha='center',fontsize=17)
        cb=fig.colorbar(im,cax=fig.add_axes([.17,.085,.66,.023]),orientation='horizontal')
        cb.set_ticks([0,50,100]);cb.set_label('Mean true-row percentage (%)',fontsize=17,labelpad=5)
        abbrev='N, normal; D, abnormal; A, abstain.' if task=='binary' else 'N, normal; P, papilledema; PP, pseudopapilledema; A, abstain.'
        caption=f'{"Binary" if task=="binary" else "Direct three-class"} eye-level confusion for the four primary model-specific pipelines. Each cell shows the arithmetic mean ± sample SD (ddof=1) of within-seed true-row percentages over seeds 17,42,2026,3407,9103; SD is in percentage points. {abbrev} All intended eyes, including structural abstentions, define each row denominator. Denominators per seed are {"20 normal and 16 abnormal" if task=="binary" else "20 normal, 8 papilledema, and 8 pseudopapilledema"}. The colour scale is fixed at 0–100%. The diagonals are failure-aware class recalls. Eye outcomes within patients and holdouts across seeds are correlated; this is a descriptive summary, not a pooled independent sample. Saved operational decisions are unchanged.'
        save(fig,f'main_{task}_confusion_mean_sd','main_confusion',task,caption,level='eye',means={m:means[m].tolist() for m in MODELS},sample_sds={m:sds[m].tolist() for m in MODELS},class_order=classes,outcome_order=classes+['Abstain'],seeds=SEEDS,sd_ddof=1,colour_limits=[0,100])
        for model in MODELS:
            for part,seeds in [('A',SEEDS[:3]),('B',SEEDS[3:])]:
                fig=plt.figure(figsize=(7,8.5))
                for i,seed in enumerate(seeds):
                    ax=fig.add_axes([.15,.665-i*.245,.75,.18])
                    c=source.COUNTS[task,model,seed]
                    im,_=matrix_axis(ax,100*c/c.sum(axis=1,keepdims=True),task,counts=c)
                    if i < len(seeds)-1:
                        ax.tick_params(axis='x',labelbottom=False)
                    ax.set_title(f'Seed {seed}',loc='left',fontweight='bold',fontsize=17,pad=10)
                fig.suptitle(f'{DISPLAY[model]} | {"Binary" if task=="binary" else "Three-class"}',y=.975,fontsize=20,fontweight='bold')
                fig.text(.5,.92,'Eye-level confusion',ha='center',fontsize=17)
                cb=fig.colorbar(im,cax=fig.add_axes([.19,.075,.62,.018]),orientation='horizontal')
                cb.set_ticks([0,50,100]);cb.set_label('True-row percentage (%)',fontsize=17,labelpad=5)
                save(fig,f'supp_{task}_confusion_{model}_{part}','confusion_seed',task,f'{DISPLAY[model]} {task} eye-level confusion, page {part}. Panels are seeds {seeds}, arranged in a three-row, one-column grid; the third slot on page B is blank. Cells report within-seed true-row percentages and saved eye counts (n). {abbrev} Every row includes all intended eyes, including structural abstentions. Each seed has {"20 normal and 16 abnormal" if task=="binary" else "20 normal, 8 papilledema, and 8 pseudopapilledema"} eyes. The shared colour scale is 0–100%. Primary model-specific calibrated operational decisions are used unchanged; no eyes or seeds are pooled.',model=model,level='eye',part=part,seeds=seeds,layout='3 rows x 1 column',empty_slots=3-len(seeds))


def style(ax,xlabel,ylabel):
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel=xlabel,ylabel=ylabel)
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    ax.grid(True,color='#DDE3E8',lw=.8);ax.spines[['top','right']].set_visible(False)


def legend(fig,y=.012,markers=False):
    handles=[Line2D([0],[0],color=COLOURS[i],ls=STYLES[i],marker=MARKERS[i] if markers else None,lw=2.2,label=str(s)) for i,s in enumerate(SEEDS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=5,frameon=False,fontsize=17,title='Seed',title_fontsize=17,columnspacing=.7,handlelength=1.15,handletextpad=.4)


def discrimination(binary_metrics,classwise):
    for task in ['binary','threeclass']:
        level='eye' if task=='binary' else 'patient'
        for model in MODELS:
            for ci in ([1] if task=='binary' else range(3)):
                cname='Abnormal' if task=='binary' else NAMES[ci]
                fig,axes=plt.subplots(2,1,figsize=(7,8.5))
                fig.subplots_adjust(left=.16,right=.97,top=.82,bottom=.205,hspace=.52)
                for i,seed in enumerate(SEEDS):
                    pred=source.PREDICTIONS[task,model,seed]
                    valid=pred.loc[pred.evaluable] if task=='binary' else pred.loc[~pred.abstained]
                    y=valid.label.to_numpy(int) if task=='binary' else (valid.true_label==ci).to_numpy(int)
                    p=valid.probability.to_numpy(float) if task=='binary' else valid[PROBCOLS[ci]].to_numpy(float)
                    fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
                    auc,ap=roc_auc_score(y,p),average_precision_score(y,p)
                    axes[0].plot(fpr,tpr,color=COLOURS[i],ls=STYLES[i],lw=2)
                    axes[1].plot(recall,precision,color=COLOURS[i],ls=STYLES[i],lw=2)
                    CURVE_STATS.append({'task':task,'level':level,'model':model,'seed':seed,'class':cname,'n_intended':len(pred),'n_evaluable':len(valid),'n_positive':int(y.sum()),'n_negative':int(len(y)-y.sum()),'auroc':float(auc),'average_precision':float(ap)})
                    for ctype,xv,yv in [('ROC',fpr,tpr),('PR',recall,precision)]:
                        CURVES.extend({'task':task,'level':level,'model':model,'seed':seed,'class':cname,'curve':ctype,'point':j,'x':float(a),'y':float(b)} for j,(a,b) in enumerate(zip(xv,yv)))
                    if task=='binary':
                        m=binary_metrics.query('model == @model and seed == @seed').iloc[0];expected=[m['conditional.auroc'],m['conditional.average_precision']]
                    else:
                        m=classwise.query('model == @model and seed == @seed and class_label == @ci').iloc[0];expected=[m.ovr_auroc,m.ovr_average_precision]
                    check(f'{task}_{level}_discrimination_{model}_{seed}_{ci}',np.allclose([auc,ap],expected,atol=1e-9))
                axes[0].plot([0,1],[0,1],color='#777777',ls=':',lw=1.3)
                style(axes[0],'False-positive rate','Sensitivity');style(axes[1],'Recall','Precision')
                axes[0].set_title('A  ROC',loc='left',fontweight='bold',pad=8);axes[1].set_title('B  Precision–recall',loc='left',fontweight='bold',pad=8)
                fig.suptitle(f'{DISPLAY[model]}\n{cname}',y=.985,fontsize=20,fontweight='bold',linespacing=1.1)
                fig.text(.5,.872,f'{level.capitalize()} level | calibrated',ha='center',fontsize=17)
                legend(fig)
                save(fig,f'supp_{task}_roc_pr_{model}_class{ci}','roc_pr',task,f'{DISPLAY[model]} {cname} {"versus normal" if task=="binary" else "one-versus-rest"} ROC and precision–recall discrimination at {level} level, using the primary model-specific classifier and saved calibrated probabilities. Each trace is one seed among evaluable {level}s. Structural abstentions have no probability and are excluded from these conditional curves. Seeds remain separate; no pooling or refitting was performed. Seed-specific AUROC, average precision, and denominators are recorded in discrimination_statistics.csv. {"This remains the recorded patient endpoint; the separate confusion matrices use eye-level outcomes." if task=='threeclass' else 'Binary curves and confusion matrices both use eye-level outcomes.'}',model=model,class_name=cname,level=level,seeds=SEEDS,layout='2 rows x 1 column')


def reliability_risk(tables,patient_counts):
    edges=np.linspace(0,1,6)
    for model in MODELS:
        fig,axes=plt.subplots(3,1,figsize=(7,8.5))
        fig.subplots_adjust(left=.17,right=.97,top=.85,bottom=.21,hspace=.70)
        for c,ax in enumerate(axes):
            for i,seed in enumerate(SEEDS):
                d=source.PREDICTIONS['threeclass',model,seed];d=d.loc[~d.abstained]
                p=d[PROBCOLS[c]].to_numpy(float);y=(d.true_label==c).to_numpy(int)
                bins=np.clip(np.searchsorted(edges,p,side='right')-1,0,4);xs=[];ys=[]
                for b in range(5):
                    keep=bins==b
                    if keep.any():
                        xs.append(p[keep].mean());ys.append(y[keep].mean())
                        RELIABILITY.append({'level':'patient','model':model,'seed':seed,'class':NAMES[c],'bin':b,'bin_lower':edges[b],'bin_upper':edges[b+1],'n':int(keep.sum()),'mean_probability':float(xs[-1]),'observed_frequency':float(ys[-1])})
                ax.plot(xs,ys,color=COLOURS[i],ls=STYLES[i],marker=MARKERS[i],ms=5,lw=1.8)
            ax.plot([0,1],[0,1],color='#777777',ls=':',lw=1.3)
            style(ax,'Mean probability' if c==2 else '', 'Observed')
            ax.set_title(f'{chr(65+c)}  {NAMES[c]}',loc='left',fontweight='bold',pad=8)
        fig.suptitle(f'{DISPLAY[model]} | Reliability',fontsize=20,fontweight='bold',y=.978)
        fig.text(.5,.915,'Patient level | calibrated',ha='center',fontsize=17)
        legend(fig)
        save(fig,f'supp_threeclass_reliability_{model}','reliability','threeclass',f'{DISPLAY[model]} patient-level classwise reliability, arranged in three rows: normal, papilledema, and pseudopapilledema. The vertical axis is observed class frequency and the horizontal axis is mean predicted class probability. Five fixed-width bins [0,0.2),[0.2,0.4),[0.4,0.6),[0.6,0.8),[0.8,1] match the existing saved-result definition; empty bins are omitted. Each point is computed within seed from saved calibrated probabilities among evaluable patients. The diagonal denotes perfect calibration. This preserves the recorded patient endpoint; it is distinct from the eye-level confusion matrices. Exact bin counts appear in reliability_points.csv.',model=model,level='patient',seeds=SEEDS,layout='3 rows x 1 column',bin_edges=edges.tolist())
        fig,ax=plt.subplots(figsize=(7,6.5));fig.subplots_adjust(left=.17,right=.97,top=.79,bottom=.26)
        for i,seed in enumerate(SEEDS):
            d=tables['risk_coverage.csv'].query('model == @model and seed == @seed').sort_values('rank')
            pred=source.PREDICTIONS['threeclass',model,seed]
            check(f'patient_risk_denominator_{model}_{seed}',len(d)==len(pred) and np.allclose(d.coverage,np.arange(1,len(pred)+1)/len(pred)))
            cm=patient_counts['threeclass',model,seed]
            check(f'patient_risk_terminal_{model}_{seed}',np.isclose(d.selective_risk.iloc[-1],1-np.trace(cm[:,:3])/cm.sum()))
            ax.step(d.coverage,d.selective_risk,where='post',color=COLOURS[i],ls=STYLES[i],lw=2)
        style(ax,'Coverage','Selective error rate')
        fig.suptitle(f'{DISPLAY[model]} | Risk–coverage',y=.97,fontsize=20,fontweight='bold')
        fig.text(.5,.87,'Patient level | failure-aware',ha='center',fontsize=17)
        legend(fig,y=.01)
        save(fig,f'supp_threeclass_risk_coverage_{model}','risk_coverage','threeclass',f'{DISPLAY[model]} recorded patient-level risk–coverage. Within each seed, patients are ranked by maximum calibrated class probability. Structural abstentions are appended at lowest confidence and counted as errors; coverage divides retained patients by all intended patients. Five seed-specific curves remain separate. Terminal risk equals one minus failure-aware patient accuracy. The original overall three-class patient-error definition is preserved; these curves are not eye-level or class-specific.',model=model,level='patient',seeds=SEEDS,layout='1 panel',confidence_definition='maximum_calibrated_patient_probability')


def performance(binary,patients):
    for task,table in [('binary',binary),('threeclass',patients)]:
        level='eye' if task=='binary' else 'patient'
        fig,axes=plt.subplots(2,1,figsize=(7,8.5));fig.subplots_adjust(left=.18,right=.97,top=.85,bottom=.205,hspace=.46)
        for ax,field,title in zip(axes,['failure_aware_balanced_accuracy','coverage'],['A  Failure-aware balanced accuracy','B  Coverage']):
            for j,model in enumerate(MODELS):
                group=table.query('model == @model').set_index('seed').loc[SEEDS];v=100*group[field].to_numpy(float)
                for i,val in enumerate(v):ax.scatter(j+(i-2)*.055,val,color=COLOURS[i],marker=MARKERS[i],s=56,zorder=3)
                ax.plot([j-.17,j+.17],[v.mean()]*2,color='#111111',lw=2.5)
                ax.plot([j,j],[v.min(),v.max()],color='#777777',lw=1.3)
            ax.set(ylim=(0,104),ylabel='Percent (%)');ax.set_yticks([0,50,100])
            ax.set_xticks(range(4),['YOLO','ViT','EMCAD','SAM2'])
            ax.set_title(title,loc='left',fontsize=17,fontweight='bold',pad=8)
            ax.grid(axis='y',color='#DDE3E8',lw=.8);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'{"Binary" if task=="binary" else "Three-class"} performance',y=.975,fontsize=20,fontweight='bold')
        fig.text(.5,.916,f'{level.capitalize()} level | model-specific',ha='center',fontsize=17)
        legend(fig)
        save(fig,f'supp_{task}_failure_aware_ba_coverage','supplement_performance',task,f'{"Binary eye" if task=="binary" else "Three-class patient"}-level primary model-specific performance. Points are the five seed results, horizontal black bars their arithmetic mean, and vertical grey lines the observed minimum–maximum range, not confidence intervals. Seed holdouts overlap and summaries are descriptive. Abstentions count as failures in balanced accuracy. YOLO, YOLO26; ViT, ViT-Method2; SAM2, SAM2-UNet. {"The three-class endpoint here remains patient-level, unlike the revised eye-level confusion matrices." if task=='threeclass' else "Both binary performance and confusion matrices use eye-level outcomes."}',level=level,seeds=SEEDS,layout='2 rows x 1 column')


def main_risk(tables):
    fig,axes=plt.subplots(2,2,figsize=(7,8.5))
    fig.subplots_adjust(left=.13,right=.97,top=.83,bottom=.23,hspace=.61,wspace=.48)
    for ax,model in zip(axes.flat,MODELS):
        for i,seed in enumerate(SEEDS):
            d=tables['risk_coverage.csv'].query('model == @model and seed == @seed').sort_values('rank')
            ax.step(d.coverage,d.selective_risk,where='post',color=COLOURS[i],ls=STYLES[i],lw=1.8)
        style(ax,'Coverage','Risk')
        ax.set_title(DISPLAY[model],loc='left',fontweight='bold',fontsize=17,pad=10)
    fig.suptitle('Three-class risk–coverage',y=.978,fontsize=20,fontweight='bold')
    fig.text(.5,.91,'Patient level | failure-aware',ha='center',fontsize=17)
    legend(fig,y=.03)
    save(fig,'main_threeclass_patient_risk_coverage','main_risk_coverage','threeclass',
         'Failure-aware risk–coverage for the primary three-class patient endpoint, shown for each primary model-specific pipeline. Five curves retain the five individual held-out seeds. Within a seed, patients are ordered by maximum saved calibrated class probability. Structural abstentions are appended at lowest confidence and counted as errors. Coverage is the fraction of all intended patients retained; risk is the fraction incorrect or abstained among those retained. Terminal risk equals one minus failure-aware patient accuracy. This figure preserves the recorded patient-level selective behavior and does not share the eye-level endpoint of the confusion matrices. Overlapping holdouts remain separate, and no new decision threshold was selected.',level='patient',seeds=SEEDS,layout='2 rows x 2 columns',confidence_definition='maximum_calibrated_patient_probability')


def finish():
    for name,rows in [('confusion_denominators',DENOMS),('confusion_cells_by_seed',CELL_ROWS),('discrimination_statistics',CURVE_STATS),('discrimination_curve_points',CURVES),('reliability_points',RELIABILITY)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    summary=[]
    for task in ['binary','threeclass']:
        for model in MODELS:
            v=np.stack([100*source.COUNTS[task,model,s]/source.COUNTS[task,model,s].sum(axis=1,keepdims=True) for s in SEEDS])
            for (r,c),mean in np.ndenumerate(v.mean(axis=0)):
                summary.append({'task':task,'level':'eye','model':model,'true_label':r,'predicted_label':c,'mean_percentage':float(mean),'sample_sd_percentage_points':float(v[:,r,c].std(ddof=1)),'n_seeds':5,'ddof':1})
    pd.DataFrame(summary).to_csv(OUT/'confusion_mean_sd.csv',index=False)
    for key,value in source.SOURCES.items():check(f'unchanged_{key}',source.digest(Path(value['path']))==value['sha256'])
    (OUT/'figure_index.json').write_text(json.dumps(FIGURES,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'font_audit.json').write_text(json.dumps({'minimum_font_pt':17,'final_pdf_required_minimum_pt':16,'embedding_instruction':'Embed at natural canvas dimensions; never scale below 16/17.','figures':FONT_AUDIT},indent=2),encoding='utf-8')
    (OUT/'audit_manifest.json').write_text(json.dumps({'sources':source.SOURCES,'consistency_tests':source.TESTS,'n_checks':len(source.TESTS),'figures':FIGURES,'scientific_units':{'binary_confusion':'eye','threeclass_confusion':'eye','binary_ROC_PR':'eye','threeclass_ROC_PR_reliability_risk_and_BA':'patient'},'aggregation':'Within-seed true-row percentages; mean and sample SD ddof=1; no pooling','font_policy':'Every visible text artist >=17pt, canvas <=7in wide and <=8.5in high'},indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'figures':len(FIGURES),'checks':len(source.TESTS),'sources':len(source.SOURCES),'minimum_font_pt':min(f['minimum_font_pt'] for f in FONT_AUDIT)}))


if __name__=='__main__':
    binary,patients,classwise,tables,eyes,patient_counts=initialize()
    record_eye_metrics(binary,eyes)
    confusion()
    discrimination(binary,classwise)
    reliability_risk(tables,patient_counts)
    performance(binary,patients)
    main_risk(tables)
    finish()

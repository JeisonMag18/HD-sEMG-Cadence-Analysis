
"""
Purpose
-------
Test whether recorded joint-angle excursion differed between the prescribed
0.50-Hz and 0.75-Hz conditions. Uses already-extracted `angulos` and `tempo_s`
from each per-recording NPZ file; no raw EMG or Welch processing is needed.

Analysis unit and estimand
-------------------------
- Task-reference angles: T1 Index, T2 Middle, T3 Ring, T4 Thumb,
  T5 Index, T6 Middle, T7 Index, T8 Index.
- Per recording: P95-P5 of the reference angle over 7-40 s (degrees).

- Per participant/task/cadence: average across valid repetitions.
- Within each subject/task: 0.75 minus 0.50 Hz, paired comparison.
- Taskwise Wilcoxon tests with Holm over eight tasks; additional subject-level
  overall average across eight tasks (exploratory, separate test).

"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATTERN = re.compile(r'^(?:Sub\d{3}__)?(Sub\d{3})_([1-8])_(05|075)_450_(\d+)$')
FINGER_INDEX = {1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 1, 8: 1}
FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Little')


def holm(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    usable = np.flatnonzero(np.isfinite(p))
    positions = usable[np.argsort(p[usable])]
    prev = 0.0
    for rank, i in enumerate(positions):
        prev = max(prev, min(1.0, (len(positions)-rank) * p[i]))
        q[i] = prev
    return q


def read_excursion(path: Path, task: int) -> tuple[float, float, float, int]:
    with np.load(path, allow_pickle=False) as z:
        if 'tempo_s' not in z.files or 'angulos' not in z.files:
            raise ValueError('Missing tempo_s/angulos columns')
        t = np.asarray(z['tempo_s'], dtype=float)
        a = np.asarray(z['angulos'], dtype=float)
    if t.ndim != 1 or a.ndim != 2 or a.shape != (len(t), 5):
        raise ValueError(f'Expected time vector and (N,5) angles, got {t.shape}, {a.shape}')
    if not np.isfinite(t).all():
        raise ValueError('Non-finite times')
    mask = (t >= 7-1e-5) & (t <= 40+1e-5)
    y = a[mask, FINGER_INDEX[task]]
    if len(y) < 60 or not np.isfinite(y).all():
        raise ValueError('Too few valid angular samples')
    p05, p95 = np.percentile(y, (5, 95))
    return float(p95-p05), float(p05), float(p95), int(len(y))


def run(cache_dir: Path, outdir: Path, max_files: int | None):
    if not cache_dir.is_dir():
        raise FileNotFoundError(f'Cache directory not found: {cache_dir}')
    files = [(p,m) for p in sorted(cache_dir.glob('*.npz'))
             if (m := PATTERN.fullmatch(p.stem))]
    if max_files: files=files[:max_files]
    if not files: raise FileNotFoundError('No compatible NPZ files found')
    rows, skipped=[], []
    for i,(path,m) in enumerate(files,1):
        subject,taskstr,cadstr,repstr=m.groups()
        task=int(taskstr); cadence=.5 if cadstr=='05' else .75
        try:
            exc, lo, hi, count=read_excursion(path,task)
            rows.append(dict(subject=subject,task=task,cadence_hz=cadence,
                             rep=int(repstr),finger=FINGERS[FINGER_INDEX[task]],
                             robust_excursion_deg=exc,p05_deg=lo,p95_deg=hi,
                             n_samples=count,cache_file=path.name))
        except Exception as e:
            skipped.append(dict(cache_file=path.name,error=str(e)))
        if i%200==0 or i==len(files): print(f'Processed {i}/{len(files)} NPZ files',flush=True)
    if not rows: raise RuntimeError('All inputs failed to load')
    outdir.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(outdir/'51_excursion_by_recording.csv',index=False)
    pd.DataFrame(skipped,columns=['cache_file','error']).to_csv(outdir/'51_skipped.csv',index=False)

    # Each participant/task/cadence has one score even when repetitions are unequal.
    subj=(df.groupby(['subject','task','cadence_hz'],as_index=False)
          .agg(excursion_deg=('robust_excursion_deg','mean'),n_reps=('rep','count')))
    subj.to_csv(outdir/'51_excursion_subject_task_cadence.csv',index=False)
    piv=subj.pivot_table(index=['subject','task'],columns='cadence_hz',
                         values='excursion_deg',aggfunc='first')
    if .5 not in piv.columns or .75 not in piv.columns:
        raise RuntimeError('Both cadences required; --max-files is for smoke tests only')
    paired=piv[[.5,.75]].dropna().rename(columns={.5:'excursion_050_deg',.75:'excursion_075_deg'})
    paired['delta_deg']=paired['excursion_075_deg']-paired['excursion_050_deg']
    paired=paired.reset_index()
    paired.to_csv(outdir/'51_excursion_paired_by_subject_task.csv',index=False)

    result=[]
    for task in range(1,9):
        d=paired.loc[paired.task==task]
        if d.empty: continue
        delta=d.delta_deg.to_numpy()
        p=float(wilcoxon(delta).pvalue) if np.any(delta!=0) else 1.0
        result.append(dict(task=f'T{task}',finger=FINGERS[FINGER_INDEX[task]],
                           n_subjects=len(d),mean_050_deg=d.excursion_050_deg.mean(),
                           mean_075_deg=d.excursion_075_deg.mean(),
                           mean_delta_deg=delta.mean(),median_delta_deg=np.median(delta),
                           p_wilcoxon=p))
    summary=pd.DataFrame(result)
    summary['p_holm_8']=holm(summary.p_wilcoxon.tolist())
    summary.to_csv(outdir/'51_excursion_by_task_comparison.csv',index=False)

    # Overall paired mean across tasks for each participant, where all 8 tasks exist.
    total=paired.groupby('subject').agg(mean_050=('excursion_050_deg','mean'),
                                        mean_075=('excursion_075_deg','mean'),
                                        n_tasks=('task','nunique'))
    total=total[total.n_tasks==8]
    if len(total):
        delta=total.mean_075-total.mean_050
        overall=dict(n_subjects=len(delta),mean_050_deg=total.mean_050.mean(),
                     mean_075_deg=total.mean_075.mean(),mean_delta_deg=delta.mean(),
                     median_delta_deg=delta.median(),
                     p_wilcoxon=float(wilcoxon(delta).pvalue) if np.any(delta!=0) else 1.0)
        pd.DataFrame([overall]).to_csv(outdir/'51_excursion_overall_subject_paired.csv',index=False)
    else:
        warnings.warn('No complete 8-task subject: overall comparison omitted')

    # Compact one-column publication figure (not required for main paper).
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.5,
                         'axes.edgecolor':'#111111','axes.labelcolor':'#111111',
                         'text.color':'#111111','xtick.color':'#111111','ytick.color':'#111111',
                         'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.4,2.45),layout='constrained')
    x=np.arange(len(summary),dtype=float)
    ax.plot(x-0.08,summary.mean_050_deg,'o',color='#2C5E91',ms=4,label='0.50 Hz')
    ax.plot(x+0.08,summary.mean_075_deg,'s',color='#D97A1B',ms=4,label='0.75 Hz')
    for xi,y0,y1 in zip(x,summary.mean_050_deg,summary.mean_075_deg):
        ax.plot((xi-.08,xi+.08),(y0,y1),lw=.7,c='#666666')
    ax.set_xticks(x,summary.task)
    ax.set_ylabel('Angular excursion, P95-P5 (degrees)')
    ax.set_xlabel('Movement task')
    ax.grid(axis='y',lw=.5,color='#DDDDDD')
    ax.legend(frameon=False,fontsize=7,ncol=2,loc='best')
    fig.savefig(outdir/'51_angular_excursion_by_task.pdf')
    fig.savefig(outdir/'51_angular_excursion_by_task.png',dpi=350)
    plt.close(fig)

    print('\n== PAIRED ROBUST ANGULAR EXCURSION (P95-P5, degrees) ==')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print(f'Files saved in: {outdir.resolve()}')
    print('NOTE: Exploratory post-hoc check; not a proof of a causal explanation for RMS.')
    if skipped: print(f'CAUTION: {len(skipped)} NPZ files skipped; inspect 51_skipped.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,
                   default=Path(__file__).resolve().parent/'resultados_ia753'/'features_welch_centrado')
    p.add_argument('--outdir',type=Path,
                   default=Path(__file__).resolve().parent/'resultados_ia753'/'51_angular_excursion_check')
    p.add_argument('--max-files',type=int,default=None)
    a=p.parse_args()
    run(a.cache,a.outdir,a.max_files)

if __name__=='__main__':
    main()

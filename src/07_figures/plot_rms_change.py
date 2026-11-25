"""Regenerate COLORED-data figures with BLACK labels and structural lines for IEEE.



Important:
- RMS uses subject-balanced percent change between means + paired-bootstrap 95% CI, matching script 18.
- CV uses per-recording CV of the 64 time-averaged channel RMS values; script 20.
- Ridge uses per-subject test metrics, not retraining.
- PSD requires raw EMG CSVs; review numeric check against the original article
  before replacing that panel (the code does NOT silently claim exact equality).
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE/'resultados_ia753'/'figures_ieee_color_data_black_axes_en'
# IMPORTANTE: CORES DOS DADOS sao preservadas. Apenas textos, molduras,
# eixos, marcas dos eixos e linhas de estrutura sao pretos.
INK = '#000000'       # structural lines + all labels
BLUE = '#2A608C'      # data color for EDC / cadence 0.50 Hz
ORANGE = '#B36A4C'    # data color for FDS / cadence 0.75 Hz
TEAL = '#34817B'
GREEN = '#628A62'
GOLD = '#AC8743'
GRAY = '#656565'      # secondary annotation/auxiliary lines
DARK = INK
GRID = '#e3e3e3'      # subtle neutral guide lines, NOT blue
RED = '#A76266'
MODELS=['RMS','RMS+MNF','RMS+MDF','RMS+MNF+MDF','MNF+MDF']
MODEL_LABEL=['RMS','RMS+MNF','RMS+MDF','RMS+MNF+MDF','MNF+MDF']
MODEL_COLORS=[BLUE,TEAL,GREEN,GOLD,'#84899B']
TASKS=['Index','Middle','Ring–little','Thumb','Index pinch','Middle pinch','Tripod','Five fingers']
FINGERS=['polegar','indicador','medio','anelar','minimo']

TASK_NAMES = {
    1: 'T1\nIndex',
    2: 'T2\nMiddle',
    3: 'T3\nRing / little',
    4: 'T4\nThumb',
    5: 'T5\nIndex\npinch',
    6: 'T6\nMiddle\npinch',
    7: 'T7\nTripod',
    8: 'T8\n5-finger\ngrasp',
}

# Scientific data colors + black axes/text; marker shapes also distinguish series
# when printed in grayscale.
STYLE = {
    'EDC': {'color': BLUE, 'marker': 'o'},
    'FDS': {'color': ORANGE, 'marker': 's'},
}


def _get_input(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_participant_pairs(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f'Cannot find participant-level RMS data:\n  {input_file}\n'
            'First run: python 18_estatistica_por_tarefa.py\n'
            'The aggregated Wilcoxon CSV alone is NOT sufficient for IQR or CIs.'
        )
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    required = {'sujeito', 'tarefa', 'cadencia_hz', 'regiao', 'descritor', 'valor'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV columns missing: {sorted(required - set(df.columns))}. '
                         f'Found: {df.columns.tolist()}')
    df = df[df['descritor'].astype(str).str.upper().eq('RMS')].copy()
    if df.empty:
        raise ValueError('No RMS rows in participant-level CSV.')
    df['tarefa'] = pd.to_numeric(df['tarefa'], errors='raise').astype(int)
    df['cadencia_hz'] = pd.to_numeric(df['cadencia_hz'], errors='raise')
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    df['regiao'] = df['regiao'].astype(str).str.upper().str.strip()
    df = df[df['regiao'].isin(STYLE) & df['tarefa'].between(1, 8)]
    if df.duplicated(['sujeito', 'tarefa', 'regiao', 'cadencia_hz']).any():
        raise ValueError('Duplicate participant/task/region/cadence rows. '
                         'Use 18_por_sujeito_condicao.csv, not 18_por_gravacao.csv.')

    pairs = df.pivot(index=['sujeito','tarefa','regiao'],
                     columns='cadencia_hz', values='valor')
    if 0.5 not in pairs.columns or 0.75 not in pairs.columns:
        raise ValueError('Both cadence columns (0.50 and 0.75) are required.')
    pairs = pairs[[0.5, 0.75]].rename(columns={0.5: 'rms_050', 0.75: 'rms_075'})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    pairs = pairs[pairs['rms_050'] > 0].reset_index()
    pairs['pct_per_subject'] = 100 * (pairs['rms_075'] / pairs['rms_050'] - 1.0)
    return pairs


def pointwise_bootstrap_ci(pairs: np.ndarray, rng: np.random.Generator,
                           n_bootstrap: int) -> tuple[float, float]:
    """Percentile paired-participant bootstrap of ratio of group means."""
    n = len(pairs)
    replicate_indices = rng.integers(low=0, high=n, size=(n_bootstrap, n))
    replicate_means = pairs[replicate_indices].mean(axis=1)
    if np.any(replicate_means[:, 0] <= 0):
        raise ValueError('Nonpositive .50 Hz mean encountered in bootstrap.')
    theta = 100.0 * (replicate_means[:, 1] / replicate_means[:, 0] - 1)
    lo, hi = np.quantile(theta, [0.025, 0.975])
    return float(lo), float(hi)


def compute_statistics(pairs: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for task in range(1, 9):
        for region in ('EDC','FDS'):
            sub = pairs[(pairs['tarefa']==task) & (pairs['regiao']==region)]
            if len(sub) < 3:
                raise ValueError(f'Only {len(sub)} paired subjects in T{task}/{region}; '
                                 'cannot estimate a defensible bootstrap CI.')
            values = sub[['rms_050','rms_075']].to_numpy(dtype=float)
            indiv = sub['pct_per_subject'].to_numpy(dtype=float)
            mu0, mu1 = values.mean(axis=0)
            theta = 100.0 * (mu1/mu0 - 1)
            lo, hi = pointwise_bootstrap_ci(values, rng, n_bootstrap)
            med = float(np.median(indiv))
            q1, q3 = np.quantile(indiv, [0.25,0.75])
            rows.append({
                'tarefa': task, 'regiao':region, 'n_sujeitos':len(sub),
                'rms_mean_050_uv':float(mu0), 'rms_mean_075_uv':float(mu1),
                'change_of_means_pct':float(theta),
                'paired_bootstrap_CI95_low_pct':lo,
                'paired_bootstrap_CI95_high_pct':hi,
                'median_individual_change_pct':med,
                'individual_change_IQR25_pct':float(q1),
                'individual_change_IQR75_pct':float(q3),
            })
    return pd.DataFrame(rows)


def verify_against_experiment_18(stats: pd.DataFrame, reference_file: Path,
                                 tolerance_pp: float = 0.05) -> None:
    """Do not silently ship a figure that disagrees with prior experiment 18."""
    if not reference_file.exists():
        print(f'NOTE: reference summary not found: {reference_file}. '
              'Figure can be generated, but no automatic cross-check was performed.')
        return
    ref = pd.read_csv(reference_file, encoding='utf-8-sig')
    required = {'descritor','regiao','tarefa','variacao_percentual'}
    if not required.issubset(ref.columns):
        print('NOTE: reference summary has unexpected columns; comparison skipped.')
        return
    ref = ref[ref['descritor'].astype(str).str.upper().eq('RMS')].copy()
    ref['regiao'] = ref['regiao'].astype(str).str.upper().str.strip()
    combined = stats.merge(ref[['regiao','tarefa','variacao_percentual']],
                           on=['regiao','tarefa'], how='left', validate='one_to_one')
    if combined['variacao_percentual'].isna().any():
        raise ValueError('Some RMS task/region combinations absent from reference CSV.')
    diff = (combined['change_of_means_pct'] - combined['variacao_percentual']).abs()
    biggest = float(diff.max())
    if biggest > tolerance_pp:
        raise ValueError(
            f'New figure does not match experiment 18 (max deviation = '
            f'{biggest:.3f} percentage points). STOP: check exclusions and aggregation.'
        )
    print(f'Cross-check passed: 16 point estimates match experiment 18 '
          f'(largest deviation {biggest:.6f} percentage points).')


def plot_point_range(stats:pd.DataFrame, output_dir:Path, kind:str) -> None:
    """Actual participant-level estimator and whiskers; NO fabricated values."""
    if kind=='mean-ci':
        y='change_of_means_pct'
        lo_col='paired_bootstrap_CI95_low_pct'
        hi_col='paired_bootstrap_CI95_high_pct'
        ylabel='Change in mean RMS (%)'
        name='37_rms_point_range_mean_CI95_en'
    elif kind=='median-iqr':
        y='median_individual_change_pct'
        lo_col='individual_change_IQR25_pct'
        hi_col='individual_change_IQR75_pct'
        ylabel='Median individual RMS change (%)'
        name='37_rms_point_range_median_IQR_en'
    else: raise ValueError(kind)
    fig,ax=plt.subplots(figsize=(3.46,2.35),layout='constrained')
    xx=np.arange(1,9,dtype=float)
    all_low=[];all_high=[]
    for region,offset in [('EDC',-.11),('FDS',.11)]:
        sub=stats[stats.regiao.eq(region)].sort_values('tarefa')
        if sub.tarefa.tolist()!=list(range(1,9)):
            raise ValueError(f'Missing task results in {region}')
        central=sub[y].to_numpy(float)
        lower=sub[lo_col].to_numpy(float)
        upper=sub[hi_col].to_numpy(float)
        if not np.all((lower<=central)&(central<=upper)):
            raise ValueError('A range does not contain the point estimate')
        st=STYLE[region]
        ax.errorbar(xx+offset,central,yerr=np.vstack([central-lower,upper-central]),
                    fmt=st['marker'], color=st['color'],ecolor=st['color'],
                    ms=3.5,mec=st['color'],mfc='white' if region=='FDS' else st['color'],
                    elinewidth=.85,capsize=1.75,capthick=.8,ls='none',
                    label=region,zorder=3)
        all_low.extend(lower);all_high.extend(upper)
    ax.set_xlim(.57,8.43)
    ymin=min(0.,float(np.min(all_low)))
    ymax=float(np.max(all_high))
    ax.set_ylim(ymin-1, ymax+max(2,(ymax-ymin)*.1))
    ax.set_xticks(xx,[f'T{i}' for i in range(1,9)])
    ax.set_xlabel('Movement task')
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2,loc='upper left',fontsize=7.6,handlelength=1.2,
              borderpad=.2,labelspacing=.2,handletextpad=.3)
    axes_simple(ax)
    save(fig,output_dir,name)


def style():
    """Colored scientific data, black axes/labels, neutral grids for publication."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'font.size': 8.3, 'axes.labelsize': 8.8,
        'xtick.labelsize': 7.7, 'ytick.labelsize': 7.7,
        'axes.edgecolor': INK, 'axes.linewidth': 0.65,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.labelcolor': INK, 'text.color': INK,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'legend.frameon': False,
        'grid.color': GRID, 'grid.linewidth': 0.45,
        'grid.linestyle': '-',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 320, 'figure.dpi': 120,
    })


def axes_simple(ax, grid=True, axis='y'):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK)
    ax.tick_params(direction='out', length=2.6, width=.6, pad=2)
    if grid: ax.grid(axis=axis, alpha=1, linewidth=.45, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, tag, name=None):
    """Panel label without an oversized subplot title."""
    lbl=tag if not name else f'{tag} {name}'
    ax.text(.01,.98,lbl, transform=ax.transAxes, ha='left', va='top',
            fontsize=8.1, fontweight='normal', color=INK,
            bbox=dict(facecolor='white',edgecolor='none',pad=1.1,alpha=.9))


def save(fig, out:Path, name:str):
    out.mkdir(parents=True,exist_ok=True)
    pdf=out/f'{name}.pdf'; png=out/f'{name}.png'
    fig.savefig(pdf,bbox_inches='tight',pad_inches=.035)
    fig.savefig(png,dpi=320,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    print('  GENERATED:',pdf)


def find_csv(name:str)->Path:
    candidates = [HERE/'resultados_ia753'/'18_analise_por_tarefa_centrado'/name,
                  HERE/'resultados_ia753'/'20_final_centrado'/name,
                  HERE/'resultados_ia753'/'26_ridge_cinco_dedos_ablacao'/name,
                  HERE/'resultados_ia753'/name, HERE/name]
    for p in candidates:
        if p.is_file(): return p
    # busca de fallback na arvore de resultados, sem atravessar dados brutos
    base=HERE/'resultados_ia753'
    if base.is_dir():
        found=list(base.rglob(name))
        if len(found)==1: return found[0]
    raise FileNotFoundError('Missing CSV: '+name+' (run the corresponding original experiment first)')

def read(name):
    return pd.read_csv(find_csv(name),encoding='utf-8-sig')

def fig_rms(out: Path):
    """Use the same subject-paired estimand as experiment 18.

    A point is the % change between condition means; confidence limits
    come from a paired participant bootstrap (10,000 resamples).
    """
    data_file = find_csv('18_por_sujeito_condicao.csv')
    ref_file = find_csv('18_resultados_wilcoxon_holm.csv')
    pairs=read_participant_pairs(data_file)
    stats=compute_statistics(pairs, n_bootstrap=10000, seed=20261120)
    verify_against_experiment_18(stats, ref_file)
    out.mkdir(parents=True,exist_ok=True)
    stats.to_csv(out/'37_rms_point_range_values.csv',index=False,encoding='utf-8-sig')
    plot_point_range(stats,out,'mean-ci')
    plot_point_range(stats,out,'median-iqr')


def fig_cv(out):
    """Per-participant mean ± 1 SD, as used in the manuscript (not 95% CI)."""
    raw=read('20_descritores_por_gravacao.csv')
    raw=raw[raw.descritor.astype(str).str.upper().eq('CV_RMS')]
    subj=raw.groupby(['sujeito','tarefa','cadencia_hz','regiao'],as_index=False)['valor'].mean()
    fig,axs=plt.subplots(2,1,figsize=(3.46,4.10),sharex=True,layout='constrained')
    x=np.arange(1,9,dtype=float)
    for ax,reg,tag in zip(axs,['EDC','FDS'],['(a)','(b)']):
        ss=subj[subj.regiao.eq(reg)]
        for cad,offset,c,mark in [(.5,-.10,BLUE,'o'),(.75,.10,ORANGE,'s')]:
            tab=ss[np.isclose(ss.cadencia_hz,cad)].groupby('tarefa').valor.agg(['mean','std']).reindex(range(1,9))
            if tab.isna().any().any(): raise ValueError('CV data incomplete')
            ax.errorbar(x+offset,tab['mean'].to_numpy(float),yerr=tab['std'].to_numpy(float),
                        fmt=mark,ms=3.2,mfc='white' if cad==.75 else c,mec=c,
                        color=c,ecolor=c,elinewidth=.7,capsize=1.5,ls='none',
                        label=f'{cad:.2f} Hz',zorder=3)
        axes_simple(ax)
        panel(ax,tag,reg)
        ax.set_ylabel('64-channel RMS CV (%)')
        ax.set_xlim(.6,8.4)
        ax.legend(ncol=2,loc='upper right',fontsize=7.0,borderpad=.1,
                  handlelength=1.2,handletextpad=.2)
    axs[-1].set_xticks(x,[f'T{i}' for i in range(1,9)])
    axs[-1].set_xlabel('Movement task')
    save(fig,out,'20_cv_ieee')


def fig_corr(out):
    """One compact 8x4 heatmap; numeric values remain readable in grayscale print."""
    zero=read('20_correlacao_tau_zero.csv')
    mat=np.full((8,4),np.nan)
    for i in range(1,9):
        for ci,(cad,reg) in enumerate([(.5,'EDC'),(.5,'FDS'),(.75,'EDC'),(.75,'FDS')]):
            ss=zero[(zero.tarefa.eq(i)) & (zero.regiao.eq(reg)) & np.isclose(zero.cadencia_hz,cad)]
            if len(ss)==1:mat[i-1,ci]=float(ss.r_mediana_sujeitos.iloc[0])
    if not np.isfinite(mat).all(): raise ValueError('Missing or duplicate heatmap values')
    fig,ax=plt.subplots(figsize=(3.46,3.60),layout='constrained')
    # This diverging COLOR scale represents signed Pearson correlation.
    im=ax.pcolormesh(np.arange(5)-.5, np.arange(9)-.5, mat,
                     cmap='RdBu_r',vmin=-1,vmax=1,edgecolors='white',linewidth=.75,shading='flat')
    ax.set_xlim(-.5,3.5)
    ax.set_ylim(7.5,-.5)
    ax.set_xticks(range(4),['EDC','FDS','EDC','FDS'])
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(8),[f'T{i}' for i in range(1,9)])
    ax.tick_params(length=0,pad=2,labelsize=7.8)
    ax.axvline(1.5,color='white',lw=2)
    for (r,c),v in np.ndenumerate(mat):
        ax.text(c,r,f'{v:+.2f}',ha='center',va='center',fontsize=7.5,
                color=INK, path_effects=[pe.withStroke(linewidth=1.05, foreground='white')])
    ax.text(.25,1.12,'0.50 Hz',transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color=INK)
    ax.text(.75,1.12,'0.75 Hz',transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color=INK)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=.050,pad=.025,shrink=.93)
    cb.set_label('Median Pearson r',fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    save(fig,out,'20_correlation_heatmap_ieee')


def fig_lag(out):
    lag=read('20_correlacao_cruzada_por_tarefa.csv')
    fig,axs=plt.subplots(2,2,figsize=(6.95,3.80),sharex=True,sharey=True,layout='constrained')
    xpanels=[(1,'EDC'),(1,'FDS'),(5,'EDC'),(5,'FDS')]
    for ax,(task,reg),tag in zip(axs.flat,xpanels,['(a)','(b)','(c)','(d)']):
        ss=lag[lag.tarefa.eq(task) & lag.regiao.eq(reg)]
        for cad,color,ls,marker in [(.5,BLUE,'-','o'),(.75,ORANGE,'--','s')]:
            d=ss[np.isclose(ss.cadencia_hz,cad)].sort_values('lag_s')
            if len(d)!=9: raise ValueError(f'Expect 9 lags in T{task}/{reg}/{cad} Hz')
            ax.plot(d.lag_s,d.r_mediana_sujeitos,color=color,ls=ls,
                    lw=1.05,marker=marker,markersize=2.7,mfc='white' if cad==.75 else color,
                    label=f'{cad:.2f} Hz')
        ax.axhline(0,color=INK,lw=.65)
        ax.axvline(0,color=INK,lw=.65,ls=':')
        ax.set_ylim(-1.02,1.02)
        axes_simple(ax)
        panel(ax,tag,f'T{task}, {reg}')
        if task==5: ax.set_xlabel('Lag (s)')
        if reg=='EDC': ax.set_ylabel('Median Pearson r')
        if task==1 and reg=='FDS': ax.legend(loc='lower left',fontsize=7.7)
    save(fig,out,'20_cross_correlation_lags_ieee')


def boxplot(ax,vals,colors,vert=False):
    bp=ax.boxplot(vals,patch_artist=True,widths=.58,showmeans=True,vert=vert,
                  meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor=INK,markersize=3),
                  medianprops=dict(color=INK,lw=1.15),
                  whiskerprops=dict(color=INK,lw=.65),
                  capprops=dict(color=INK,lw=.65),
                  flierprops=dict(marker='.',markersize=2.2,markerfacecolor=INK,markeredgecolor=INK,alpha=.7))
    for b,c in zip(bp['boxes'],colors):
        b.set(facecolor=c,edgecolor=INK,linewidth=.75,alpha=.78)
    return bp


def fig_ridge(out):
    rows=read('26_ridge_por_sujeito.csv')
    metric='R2_vw' if 'R2_vw' in rows else 'R2_vw_sujeito'
    # One-column paired horizontal boxplots; labels stay readable at IEEE size.
    fig,axs=plt.subplots(2,1,figsize=(3.46,4.60),sharex=True,layout='constrained')
    for ax,cad,tag in zip(axs,[.5,.75],['(a)','(b)']):
        ss=rows[np.isclose(rows.cadencia_hz,cad)]
        vals=[pd.to_numeric(ss[ss.modelo.eq(m)][metric],errors='coerce').dropna() for m in MODELS]
        if any(len(v)<3 for v in vals): raise ValueError('Missing Ridge participant results')
        boxplot(ax,vals,MODEL_COLORS,vert=False)
        ax.set_yticks(range(1,6),MODEL_LABEL)
        ax.set_ylim(5.6,.4)
        ax.axvline(0,color=INK,lw=.65)
        axes_simple(ax,axis='x')
        panel(ax,tag,f'{cad:.2f} Hz')
    axs[-1].set_xlabel(r'Variance-weighted $R^2$')
    save(fig,out,'26_ridge_boxplots_ieee')

    fingers=read('26_ridge_por_dedo_resumo.csv')
    metric='R2_media' if 'R2_media' in fingers else 'R2_medio'
    fig,axs=plt.subplots(2,1,figsize=(3.46,4.85),sharex=True,layout='constrained')
    arrs=[]
    for cad in [.5,.75]:
        ss=fingers[np.isclose(fingers.cadencia_hz,cad)].copy()
        ss['dedo']=ss.dedo.astype(str).str.strip().str.lower()
        arr=ss.pivot(index='modelo',columns='dedo',values=metric).reindex(index=MODELS,columns=FINGERS).to_numpy(float)
        if not np.isfinite(arr).all(): raise ValueError('Missing per-finger data')
        arrs.append(arr)
    vmin=min(0.,*(float(a.min()) for a in arrs))
    vmax=max(.85,*(float(a.max()) for a in arrs))
    for ax,arr,cad,tag in zip(axs,arrs,[.5,.75],['(a)','(b)']):
        im=ax.pcolormesh(np.arange(6)-.5, np.arange(6)-.5, arr,
                          cmap='YlGnBu',vmin=vmin,vmax=vmax,edgecolors='white',linewidth=.5,shading='flat')
        ax.set_xlim(-.5,4.5)
        ax.set_ylim(4.5,-.5)
        ax.set_yticks(range(5),MODEL_LABEL)
        ax.set_xticks(range(5),['Thumb','Index','Middle','Ring','Little'])
        ax.tick_params(length=0,pad=2,labelsize=7.0)
        for (i,j),v in np.ndenumerate(arr):
            a=(v-vmin)/(vmax-vmin) if vmax>vmin else .5
            ax.text(j,i,f'{v:.2f}',fontsize=7.1,ha='center',va='center',
                    color=INK, path_effects=[pe.withStroke(linewidth=1.05,foreground='white')])
        panel(ax,tag,f'{cad:.2f} Hz')
        for sp in ax.spines.values():sp.set_visible(False)
    cb=fig.colorbar(im,ax=axs,fraction=.035,pad=.02,shrink=.87)
    cb.set_label(r'Mean finger $R^2$',fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    save(fig,out,'26_ridge_fingers_ieee')


def fig_fluxo(out):
    """Compact vertical pipeline for ONE IEEE column; entirely vector."""
    fig,ax=plt.subplots(figsize=(3.46,3.15),layout='constrained')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    boxes=[(.82,'128-channel HD-sEMG + finger angles'),
           (.62,'RMS / ARV / Welch PSD / MNF / MDF'),
           (.42,'Cadence comparison, CV, cross-correlation'),
           (.22,'Multi-output Ridge regression'),
           (.02,'Spectral and kinematic results')]
    w=.90; h=.14; left=.05
    for y,text_ in boxes:
        ax.add_patch(FancyBboxPatch((left,y),w,h,boxstyle='square,pad=0',
                                   facecolor='white',edgecolor=INK,linewidth=.85))
        ax.text(.5,y+h/2,text_,ha='center',va='center',fontsize=7.7)
    for yhi,ylo in zip([.82,.62,.42,.22],[.62,.42,.22,.02]):
        ax.add_patch(FancyArrowPatch((.5,yhi),(.5,ylo+h),arrowstyle='-|>',
                                     mutation_scale=8,lw=.85,color=INK))
    save(fig,out,'03_workflow_ieee')


def fig_ilustracoes(out):
    """Reformat existing scientific illustrations with clean English labels.

    Anatomical source remains unmodified. For the task montage, original
    gesture photographs are kept, but old oversized italic labels are cropped
    and replaced by readable Matplotlib labels, to unify the visual style.
    """
    from PIL import Image
    anatomy_file = HERE / 'eds_fds.png'
    gestures_file = HERE / 'gestos_mao_sem_borda.png'
    for src in (anatomy_file, gestures_file):
        if not src.is_file():
            raise FileNotFoundError(f'Missing original illustration: {src}')

    img = np.asarray(Image.open(anatomy_file).convert('RGB'))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(9.0, 9.0*h/w))
    ax.imshow(img, interpolation='bilinear')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, out, '01_muscles_ieee')

    img = np.asarray(Image.open(gestures_file).convert('RGB'))
    h,w = img.shape[:2]
    # Cropping bounds are relative to the original 4-by-2 composition.
    # The old text is around 40%-48% and 92%-100% of image height.
    row_bounds = [(0.00,0.395),(0.495,0.920)]
    labels = ['T1 · Index','T2 · Middle','T3 · Ring–little','T4 · Thumb',
              'T5 · Index pinch','T6 · Middle pinch','T7 · Tripod','T8 · Five fingers']
    fig, axs = plt.subplots(2,4,figsize=(6.95,4.05),
                            gridspec_kw={'wspace':0.04,'hspace':0.10})
    for row,(y0,y1) in enumerate(row_bounds):
        for col in range(4):
            i = row*4+col
            x0=int(round(col*w/4))
            x1=int(round((col+1)*w/4))
            tile=img[int(y0*h):int(y1*h),x0:x1]
            ax=axs[row,col]
            ax.imshow(tile,interpolation='bilinear')
            ax.axis('off')
            ax.set_title('')
            ax.text(.5,-.055,labels[i],transform=ax.transAxes,
                    ha='center',va='top',fontsize=7.8,fontweight='normal',color=DARK)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.07)
    save(fig,out,'02_tasks_ieee')


def fig_psd_from_raw(out,dataset:Path,fs=2052.52):
    """Recalcula a PSD global. Validar valores numericos antes de usar no artigo.

    Welch com Hann, 1s ~2048 amostras e 50% overlap (como descricao do artigo).
    Assumido: colunas 0-63 EDC, 64-127 FDS; amplitude bruta em µV.
    Encontra CSVs com nome Sub###_tarefa_cadencia_450_repeticao.csv.
    """
    from scipy.signal import welch
    import csv
    pat=re.compile(r'^(?:Sub\d{3}__)?(Sub\d{3})_([1-8])_(05|075)_450_(\d+)$',re.I)
    files=[]
    for p in dataset.rglob('*.csv'):
        m=pat.fullmatch(p.stem)
        if not m: continue
        if re.search(r'kinematic|mocap|angulo',str(p.parent),re.I): continue
        files.append((p,m.groups()))
    if not files:
        raise FileNotFoundError(f'No recording CSVs in {dataset}; pass a valid raw EMG directory via --dataset')
    print('  PSD: candidate recordings:',len(files))
    curves=[]; failed=[]
    ns=2048
    for i,(p,(sub,task,ftxt,rep)) in enumerate(files,1):
        try:
            # Detecta cabeçalho. Os CSVs de EMG têm 128 canais numéricos.
            # O arquivo de cinemática com nome igual tem <128 colunas e é ignorado.
            with p.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
                reader=csv.reader(f)
                head=next(reader)
            ncols=len(head)
            if ncols!=128:
                continue
            try:
                float(head[0]); header=None
            except ValueError:
                header=0
            arr=pd.read_csv(p,header=header,dtype='float32').to_numpy(dtype='float32')
            if arr.shape[1]!=128: continue
            a=arr[int(round(5*fs)):int(round(40*fs))]
            if len(a)<ns: continue
            fr,ps=welch(a,fs=fs,window='hann',nperseg=ns,noverlap=ns//2,
                        axis=0,detrend='constant',scaling='density')
            mask=(fr>=10)&(fr<=500)
            f=fr[mask]
            for reg,sl in [('EDC',slice(0,64)),('FDS',slice(64,128))]:
                pxx=np.mean(ps[mask,sl],axis=1)
                area=np.trapezoid(pxx,f)
                if not np.isfinite(area) or area<=0: continue
                curves.append({'sub':sub,'task':int(task),'cad':.5 if ftxt=='05' else .75,
                               'rep':int(rep),'reg':reg,'f':f,'abs':pxx,'norm':pxx/area})
        except Exception as ex:
            failed.append((p.name,str(ex)))
        if i%100==0: print(f'  PSD: {i}/{len(files)} arquivos varridos')
    if not curves: raise RuntimeError('No raw 128-channel EMG was processed. Check the CSV format.')
    # Balanceia repeticoes, tarefas e, por fim, sujeitos.
    agg={}
    for reg in ['EDC','FDS']:
        for cad in [.5,.75]:
            group=defaultdict(list)
            for c in curves:
                if c['reg']==reg and c['cad']==cad:
                    group[(c['sub'],c['task'])].append(c)
            per_subject=defaultdict(list)
            for (sub,task),reps in group.items():
                per_subject[sub].append((np.mean([v['abs'] for v in reps],axis=0),
                                         np.mean([v['norm'] for v in reps],axis=0)))
            subjects=[]
            for sub,tsk in per_subject.items():
                subjects.append((np.mean([x for x,y in tsk],axis=0),
                                 np.mean([y for x,y in tsk],axis=0)))
            if not subjects: raise RuntimeError(f'No PSD available for {reg} {cad}')
            agg[(reg,cad)]={'n':len(subjects),
                            'abs':np.mean([z[0] for z in subjects],axis=0),
                            'norm':np.mean([z[1] for z in subjects],axis=0),
                            'abs_sd':np.std([z[0] for z in subjects],axis=0,ddof=1),
                            'norm_sd':np.std([z[1] for z in subjects],axis=0,ddof=1)}
    # Wider four-panel scientific plot: width is justified by four PSD panels.
    fig,axs=plt.subplots(2,2,figsize=(6.95,4.85),sharex=True,layout='constrained')
    for k,(reg,kind) in enumerate([('EDC','abs'),('FDS','abs'),('EDC','norm'),('FDS','norm')]):
        ax=axs.flat[k]
        for cad,col,linestyle in [(.5,BLUE,'-'),(.75,ORANGE,'--')]:
            v=agg[(reg,cad)];m=v[kind];sd=v[kind+'_sd']
            ax.plot(f,m,color=col,lw=1.15,ls=linestyle,label=f'{cad:.2f} Hz')
            if kind=='norm':
                ax.fill_between(f,np.maximum(0,m-sd),m+sd,color=col,alpha=.10,linewidth=0)
        axes_simple(ax)
        panel(ax,['(a)','(b)','(c)','(d)'][k],reg)
        if kind=='abs': ax.set_yscale('log')
        if k in (0,2):
            ax.set_ylabel(r'PSD ($\mu$V$^2$/Hz)' if kind=='abs' else 'Normalized PSD (1/Hz)')
        if k in (2,3):ax.set_xlabel('Frequency (Hz)')
        if k==1:ax.legend(loc='upper right',fontsize=7.8)
    save(fig,out,'32_psd_ieee_REVIEW_NUMBERS')
    # Salva valores que permitem verificar incompatibilidades.
    lines=[]
    # Bandpower reference values in the existing manuscript, for validation only.
    manuscript_bandpower = {('EDC',.5):6363.21, ('EDC',.75):9368.13,
                            ('FDS',.5):3849.18, ('FDS',.75):5944.39}
    for reg in ['EDC','FDS']:
        for cad in [.5,.75]:
            v=agg[(reg,cad)]
            area=float(np.trapezoid(v['abs'],f))
            expected=manuscript_bandpower[(reg,cad)]
            deviation=100*(area/expected-1)
            lines.append(f'{reg}, {cad:.2f} Hz, n={v["n"]}, area = {area:.4f} µV²; '
                         f'manuscript = {expected:.2f} µV²; difference = {deviation:+.2f}%')
            if abs(deviation)>5:
                print(f'  WARNING PSD {reg} {cad:.2f}: area differs by {deviation:+.1f}% '
                      'from the current manuscript. Do NOT silently replace the original.')
    (out/'32_PSD_NUMERIC_AUDIT.txt').write_text('\n'.join(lines)+'\n\n'+
        'BEFORE USING: compare against the original paper PSD and bandpower.\n'+
        'Numerical deviations must be explained by documented Welch choices.\n',encoding='utf-8')
    if failed:
        pd.DataFrame(failed,columns=['arquivo','erro']).to_csv(out/'32_files_with_errors.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset',type=Path,help='Directory with raw HD-sEMG CSVs, used only for PSD.')
    parser.add_argument('--output','--saida',type=Path,default=OUT_DEFAULT)
    parser.add_argument('--only','--somente',dest='only',choices=['all','rms','cv','correlation','correlation-map','correlation-lag','ridge','psd','flow','illustrations'],default='all')
    args=parser.parse_args()
    out=args.output
    style()
    works=[('rms',lambda:fig_rms(out)),('cv',lambda:fig_cv(out)),
           ('correlation-map',lambda:fig_corr(out)),('correlation-lag',lambda:fig_lag(out)),
           ('ridge',lambda:fig_ridge(out)),
           ('flow',lambda:fig_fluxo(out)),('illustrations',lambda:fig_ilustracoes(out))]
    if args.only in ('all','psd'):
        if args.dataset:
            works.append(('psd',lambda:fig_psd_from_raw(out,args.dataset)))
        else:
            print('  PSD NOT GENERATED: pass --dataset path/to/raw_EMG_CSVs. No approximation is substituted.')
    errors=[]
    for name,func in works:
        if args.only not in ('all',name) and not (args.only=='correlation' and name.startswith('correlation-')): continue
        try:
            print(f'\n[{name}]')
            func()
        except Exception as exc:
            print(f'  SKIPPED: {name}: {type(exc).__name__}: {exc}')
            errors.append((name,str(exc)))
    print('\nOutput folder:',out)
    if errors:
        print('Some figures were skipped. Check the source CSV path and run experiments 18/20/26.')

if __name__=='__main__':
    main()

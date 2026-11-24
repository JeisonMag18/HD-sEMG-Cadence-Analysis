#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utiliza os resultados CSV produzidos por ridge_feature_comparison.py,
sem treinar novamente os modelos ou processar os sinais HD-sEMG.

Saídas:
    - Boxplots dos erros MAE dos seis modelos (Figura 9).
    - Mapa de calor do R² por dedo.
    - Tabela LaTeX da regressão Ridge (Tabela VII).
    - Tabela de correlações alinhadas da tarefa T5, quando disponível.
    - Textos auxiliares de metodologia e resultados.
"""



from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE = '#2C5E91'
ORANGE = '#D97A1B'
BLACK = '#111111'
GRID = '#E3E3E3'
MODELS = ('RMS', 'RMS+MNF', 'RMS+MDF', 'RMS+MNF+MDF', 'MNF+MDF', 'RMS+RMS_context')
FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Little')
LATEX_LABELS = {
    'RMS': 'RMS',
    'RMS+MNF': 'RMS+MNF',
    'RMS+MDF': 'RMS+MDF',
    'RMS+MNF+MDF': 'RMS+MNF+MDF',
    'MNF+MDF': 'MNF+MDF',
    'RMS+RMS_context': 'RMS+Smoothed RMS',
}
SHORT_LABELS = {
    'RMS': 'RMS',
    'RMS+MNF': 'RMS + MNF',
    'RMS+MDF': 'RMS + MDF',
    'RMS+MNF+MDF': 'RMS + MNF + MDF',
    'MNF+MDF': 'MNF + MDF',
    'RMS+RMS_context': 'RMS + Smoothed\nRMS (control)',
}


def styling():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'font.size': 7.7, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': BLACK,
        'axes.labelcolor': BLACK, 'xtick.color': BLACK, 'ytick.color': BLACK,
        'text.color': BLACK, 'axes.linewidth': .75, 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white',
    })


def save(fig, directory: Path, name: str):
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight', pad_inches=.04)
    fig.savefig(directory / f'{name}.png', dpi=350, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def validate(detail: pd.DataFrame, summary: pd.DataFrame, fingers: pd.DataFrame):
    for frame, required in (
        (detail, ('cadence_hz', 'subject', 'model', 'MAE_deg', 'R2_vw')),
        (summary, ('cadence_hz', 'model', 'n_subjects', 'MAE_mean', 'R2_vw_mean')),
        (fingers, ('cadence_hz', 'model', 'finger', 'R2_mean')),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
    for cadence in (.5, .75):
        s = summary[np.isclose(summary.cadence_hz, cadence)]
        if set(s.model) != set(MODELS):
            raise ValueError(f'Incomplete model summary at {cadence} Hz: {set(MODELS) - set(s.model)}')
        for model in MODELS:
            if len(detail[np.isclose(detail.cadence_hz, cadence) & (detail.model == model)]) != int(
                s.loc[s.model == model, 'n_subjects'].iloc[0]
            ):
                raise ValueError(f'Subject count mismatch: {cadence} {model}')


def plot_boxplots(detail: pd.DataFrame, directory: Path):
    """Reproduce the final IEEE Fig. 10: five ablations plus separate control."""
    styling()
    fig, axes = plt.subplots(2, 1, figsize=(3.52, 5.35), sharex=True)
    fig.subplots_adjust(left=.36, right=.985, top=.966, bottom=.086, hspace=.29)
    for i, cadence in enumerate((.5, .75)):
        ax = axes[i]
        positions = np.arange(1, len(MODELS) + 1)
        values = [
            detail.loc[np.isclose(detail.cadence_hz, cadence) & (detail.model == model),
                       'MAE_deg'].to_numpy() for model in MODELS
        ]
        primary = BLUE if i == 0 else ORANGE
        boxes = ax.boxplot(
            values, positions=positions, vert=False, patch_artist=True,
            widths=.53, showmeans=True,
            meanprops=dict(marker='D', markersize=3.7, markeredgecolor=BLACK,
                           markerfacecolor=BLACK),
            medianprops=dict(color=BLACK, linewidth=1.0),
            whiskerprops=dict(color=BLACK, linewidth=.75),
            capprops=dict(color=BLACK, linewidth=.75),
            flierprops=dict(marker='o', markersize=2.5, markeredgecolor=BLACK,
                            markerfacecolor='none'),
        )
        for j, patch in enumerate(boxes['boxes']):
            patch.set_facecolor('#D4D4D4' if j in (0, 5) else primary)
            patch.set_alpha(.94 if j in (0, 5) else .54)
            patch.set_edgecolor(BLACK)
            patch.set_linewidth(.7)
        ax.set_yticks(positions)
        ax.set_yticklabels([SHORT_LABELS[m] for m in MODELS], fontsize=7.1)
        ax.invert_yaxis()
        ax.axhline(5.5, color='#777777', linewidth=.65, linestyle=(0, (3, 3)))
        ax.grid(axis='x', color=GRID, linewidth=.55)
        ax.set_axisbelow(True)
        ax.set_title(f'({"a" if i == 0 else "b"}) {cadence:.2f} Hz',
                     fontsize=8.5, color=BLACK, loc='left', pad=4)
        ax.tick_params(axis='both', length=3, width=.65, colors=BLACK, labelsize=7.1)
        for spine in ax.spines.values():
            spine.set_color(BLACK)
            spine.set_linewidth(.75)
    axes[-1].set_xlabel('Test MAE (degrees)')
    save(fig, directory, '49_ridge_MAE_boxplots_single_column')


def plot_fingers(fingers: pd.DataFrame, directory: Path):
    styling()
    fig, axes = plt.subplots(2, 1, figsize=(3.48, 5.2))
    fig.subplots_adjust(left=.35, right=.87, top=.95, bottom=.105, hspace=.28)
    palette = LinearSegmentedColormap.from_list('scientific_soft', ['#A7C4DA', '#FFFFFF', '#F0B985'])
    vmin = min(-1., float(fingers.R2_mean.min()))
    vmax = max(1., float(fingers.R2_mean.max()))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    last_img = None
    for i, cadence in enumerate((.5, .75)):
        ax = axes[i]
        f = fingers[np.isclose(fingers.cadence_hz, cadence)]
        pivot = f.pivot(index='model', columns='finger', values='R2_mean').reindex(
            index=MODELS, columns=FINGERS)
        if pivot.isna().any().any():
            raise ValueError(f'Missing finger R2 for cadence {cadence}')
        matrix = pivot.to_numpy()
        last_img = ax.imshow(matrix, aspect='auto', cmap=palette, norm=norm,
                             interpolation='nearest')
        ax.set_yticks(np.arange(len(MODELS)), [SHORT_LABELS[m] for m in MODELS], fontsize=6.5)
        ax.axhline(4.5, color=BLACK, linewidth=.65, linestyle=(0, (3, 3)))
        ax.set_xticks(np.arange(5), FINGERS, fontsize=6.4)
        ax.tick_params(length=0, colors=BLACK)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.0)
        ax.tick_params(which='minor', bottom=False, left=False)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f'{matrix[row, col]:.2f}', ha='center', va='center',
                        color=BLACK, fontsize=6.5)
        ax.set_title(f'({"a" if i == 0 else "b"}) {cadence:.2f} Hz',
                     fontsize=8.6, loc='left', pad=4, color=BLACK)
        for spine in ax.spines.values(): spine.set_color(BLACK)
    cax = fig.add_axes([.91, .14, .024, .7])
    cb = fig.colorbar(last_img, cax=cax)
    cb.ax.tick_params(labelsize=6.4, colors=BLACK)
    cb.set_label('Mean finger R²', fontsize=7, color=BLACK)
    save(fig, directory, '49_ridge_finger_R2_heatmap_single_column')



def results_text(summary: pd.DataFrame, stats: pd.DataFrame) -> str:
    def value(cad, model, key):
        row = summary[np.isclose(summary.cadence_hz, cad) & (summary.model == model)]
        return float(row.iloc[0][key])
    def pstr(cad, model, ref):
        f = stats[np.isclose(stats.cadence_hz, cad) & (stats.model == model) &
                  (stats.reference == ref)]
        return (f'{float(f.iloc[0].p_holm):.3g}' if len(f) and np.isfinite(float(f.iloc[0].p_holm)) else 'not available')
    lines = [
        '% Generated from Experiment 48 CSV. Insert into Ridge Results; verify full manuscript consistency.',
        r'\subsection{Aligned Ridge Regression and Temporal-Context Control}',
    ]
    for cad in (.5, .75):
        lines.append(
            f'At {cad:.2f}~Hz, the mean test MAE for RMS alone was '
            f'{value(cad,"RMS","MAE_mean"):.2f}$^\\circ$, compared with '
            f'{value(cad,"RMS+MNF","MAE_mean"):.2f}$^\\circ$ for aligned RMS+MNF '
            f'and {value(cad,"RMS+RMS_context","MAE_mean"):.2f}$^\\circ$ for '
            f'the RMS+Smoothed RMS control. The corresponding mean $R^2_{{vw}}$ values '
            f'were {value(cad,"RMS","R2_vw_mean"):.4f}, '
            f'{value(cad,"RMS+MNF","R2_vw_mean"):.4f}, and '
            f'{value(cad,"RMS+RMS_context","R2_vw_mean"):.4f}, respectively. '
            f'The Holm-adjusted two-sided Wilcoxon $p$ value for aligned '
            f'RMS+MNF versus the temporal control was '
            f'{pstr(cad,"RMS+MNF","RMS+RMS_context")}.\n'
        )
    lines.append('All models were evaluated at the same trimmed time points. '
                 'These estimates therefore should not be equated numerically '
                 'with the earlier unaligned ablation.')
    return '\n\n'.join(lines) + '\n'



def main():
    p = argparse.ArgumentParser(
        description="Gera figuras da regressão Ridge a partir dos CSVs."
    )

    p.add_argument(
        "--base",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "02_features"
            / "resultados_ia753"
        ),
        help="Diretório dos resultados da análise"
    )

    p.add_argument(
        "--allow-incomplete-test",
        action="store_true",
        help="Permite testar com menos de 21 participantes"
    )

    args = p.parse_args()

    exp = args.base / "48_ridge_full_aligned_ablation"

    detail = pd.read_csv(exp / "48_ridge_aligned_by_subject.csv")
    summary = pd.read_csv(exp / "48_ridge_aligned_summary.csv")
    fingers = pd.read_csv(exp / "48_ridge_aligned_finger_summary.csv")

    validate(detail, summary, fingers)

    nvalues = set(summary["n_subjects"].astype(int))

    if nvalues != {21} and not args.allow_incomplete_test:
        raise RuntimeError(
            "A análise precisa conter os 21 participantes."
        )

    out_name = (
        "49_aligned_paper_outputs"
        if nvalues == {21}
        else "49_aligned_paper_outputs_TEST_ONLY"
    )

    out = args.base / out_name
    out.mkdir(parents=True, exist_ok=True)

    # Geração das figuras
    plot_boxplots(detail, out)
    plot_fingers(fingers, out)

    print("Figures generated successfully in:", out.resolve())


if __name__ == "__main__":
    main()

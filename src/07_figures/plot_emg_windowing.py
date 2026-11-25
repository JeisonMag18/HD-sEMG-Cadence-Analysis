"""
Gera a figura de alinhamento temporal das janelas HD-sEMG.

Painéis:
    (a) Janela espectral de 2 s e RMS centrado.
    (b) Janela RMS de 617 amostras.
    (c) Segmentos Hann com 50% de sobreposição.

Utiliza uma gravação real do dataset.
Exporta a figura em PNG e PDF.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

FEATURES_DIR = Path(__file__).resolve().parents[1] / "02_features"
sys.path.insert(0, str(FEATURES_DIR))

FS = 2052.52
BLUE = '#2C5E91'
ORANGE = '#D97A1B'
BLACK = '#171717'
GRID = '#DFDFDF'


def find_example_emg() -> Path:
    """Uses the user's existing indexer to locate ONE real recording."""
    try:
        from features import indexar_gravacoes
        recordings, _notes = indexar_gravacoes()
    except Exception as exc:
        raise FileNotFoundError(
            'Unable to auto-locate EMG. Pass --emg with the full path to one raw 128-channel CSV. '
            f'Details: {exc}'
        ) from exc
    if not recordings:
        raise FileNotFoundError('Indexer returned no recordings. Use --emg "path_to_file.csv".')
    preferred = 'Sub001_1_05_450_0'
    chosen = next(
        (g for g in recordings if str(getattr(g, 'id', '')).endswith(preferred)
         or Path(g.emg).stem.endswith(preferred)),
        recordings[0],
    )
    return Path(chosen.emg)


def read_emg_channel(path: Path, channel_zero_based: int) -> np.ndarray:
    """Load a single numeric EMG column, supporting optional header and optional time column."""
    first_rows = pd.read_csv(path, header=None, nrows=4, low_memory=False)
    ncols = first_rows.shape[1]
    if ncols not in (128, 129):
        raise ValueError(
            f'Expected 128 EMG columns (or 129 with a time column), got {ncols}. '
            'Check that you selected a raw EMG file, not a kinematics CSV.'
        )
    emg_column = channel_zero_based + (1 if ncols == 129 else 0)
    first_row = pd.to_numeric(first_rows.iloc[0], errors='coerce')
    has_header = float(first_row.notna().mean()) < 0.9
    arr = pd.read_csv(
        path,
        header=0 if has_header else None,
        usecols=[emg_column],
        low_memory=False,
    ).iloc[:, 0]
    x = pd.to_numeric(arr, errors='coerce').to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('Selected EMG channel includes nonnumeric, NaN, or infinite values.')
    if np.std(x) < 1e-10:
        raise ValueError(
            'Selected channel is constant. Choose a different channel, '
            'e.g. --channel 1 or --channel 20.'
        )
    return x


def style_all():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 7.5,
        'axes.labelsize': 8.0,
        'xtick.labelsize': 7.2,
        'ytick.labelsize': 7.2,
        'legend.fontsize': 7.0,
        'axes.linewidth': 0.75,
        'axes.edgecolor': BLACK,
        'axes.labelcolor': BLACK,
        'xtick.color': BLACK,
        'ytick.color': BLACK,
        'text.color': BLACK,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


def draw_figure(
    x: np.ndarray,
    *,
    analysis_time: float,
    region: str,
    channel: int,
    outdir: Path,
    nperseg: int,
    overlap_percent: float,
    demo: bool,
    source: Path,
):
    fs = FS
    k = int(round(analysis_time * fs))
    rms_start, rms_end = k - 308, k + 309  # stop is exclusive: 617 samples
    # Spectral feature at cached index i+4: original retrospective window
    # ending ~1 s after the RMS reference time, i.e. centered near t.
    aligned_reference_k = int(round((analysis_time + 1.0) * fs))
    spectral_end = aligned_reference_k
    spectral_start = spectral_end - int(round(2.0 * fs))
    if spectral_start < 0 or spectral_end > len(x) or rms_end > len(x):
        raise ValueError('The signal is too short for these sample windows at the selected time.')
    if not (0 < nperseg <= spectral_end - spectral_start):
        raise ValueError('Invalid Welch subsegment length for this 2-s interval.')
    noverlap = int(round(nperseg * overlap_percent / 100))
    if noverlap >= nperseg:
        raise ValueError('Overlap must be less than 100%.')
    stride = nperseg - noverlap
    starts = list(range(spectral_start, spectral_end - nperseg + 1, stride))
    if not starts:
        raise ValueError('No full Welch segments fit into the spectral interval.')

    rms_value = float(np.sqrt(np.mean(np.square(x[rms_start:rms_end]))))
    spectral_duration = (spectral_end - spectral_start) / fs
    rms_duration = (rms_end - rms_start) / fs

    style_all()
    fig = plt.figure(figsize=(3.46, 5.05))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.10, 0.85],
                          left=0.19, right=0.975, top=0.90, bottom=0.11, hspace=0.63)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Three panels share the same actual source signal and the same analysis time.
    def waveform(ax, start, end):
        a = max(0, int(round(start * fs)))
        b = min(len(x), int(round(end * fs)))
        ts = np.arange(a, b, dtype=float) / fs
        ax.plot(ts, x[a:b], color=BLUE, lw=0.47, rasterized=False)
        ax.set_xlim(start, end)
        ax.set_ylabel('HD-sEMG (µV)', labelpad=2)
        ax.grid(axis='y', color=GRID, lw=0.42)
        ax.axhline(0, color=BLACK, alpha=0.38, lw=0.42)
        ax.tick_params(axis='both', length=2.2)

    # (a) Raw HD-sEMG and the centered/aligned windows.
    waveform(ax1, analysis_time - 1.25, analysis_time + 1.25)
    ax1.axvspan(spectral_start / fs, spectral_end / fs, fc=BLUE, alpha=0.11, zorder=0)
    ax1.axvspan(rms_start / fs, rms_end / fs, fc=ORANGE, alpha=0.27, zorder=0)
    ax1.axvline(analysis_time, color=BLACK, ls=(0, (3, 2)), lw=0.85)
    ax1.text(-0.14, 1.035, '(a)', transform=ax1.transAxes, weight='bold',
             ha='left', va='bottom', clip_on=False)
    ax1.set_xlabel('Time (s)', labelpad=2)

    legend = [
        Patch(facecolor=BLUE, alpha=0.16, edgecolor='none', label='Aligned spectral: 2 s'),
        Patch(facecolor=ORANGE, alpha=0.33, edgecolor='none', label='RMS: 300.6 ms'),
        Line2D([0], [0], color=BLACK, ls=(0, (3, 2)), lw=0.85, label=r'Analysis $t_k$'),
    ]
    fig.legend(handles=legend, frameon=False, ncol=3,
               loc='upper center', bbox_to_anchor=(0.57, 0.98),
               borderaxespad=0, columnspacing=0.65, handlelength=1.3,
               handletextpad=0.4, fontsize=6.6)

    # (b) Exact centered RMS: same source channel; 617 samples.
    zoom_half_span = 0.28
    waveform(ax2, analysis_time - zoom_half_span, analysis_time + zoom_half_span)
    ax2.axvspan(rms_start / fs, rms_end / fs, fc=ORANGE, alpha=0.28, zorder=0)
    ax2.axvline(analysis_time, color=BLACK, ls=(0, (3, 2)), lw=0.8)
    ax2.set_xlabel('Time (s)', labelpad=2)
    ax2.text(-0.14, 1.035, '(b)', transform=ax2.transAxes, weight='bold',
             ha='left', va='bottom', clip_on=False)
    ax2.text(0.99, 1.015, f'617 samples · {rms_duration*1000:.1f} ms',
             transform=ax2.transAxes, fontsize=6.8,
             ha='right', va='bottom', clip_on=False)

    # (c) Location of overlapping Hann segments within the 2-s aligned window.
    nseg = len(starts)
    ax3.set_xlim(spectral_start / fs - 0.1, spectral_end / fs + 0.1)
    ax3.set_ylim(-0.55, nseg - 0.25)
    ax3.set_xlabel('Time (s)', labelpad=2)
    ax3.set_yticks([])
    ax3.grid(axis='x', color=GRID, lw=0.42)
    ax3.axvline(analysis_time, color=BLACK, ls=(0, (3, 2)), lw=0.8)
    ax3.axvspan(spectral_start / fs, spectral_end / fs, color=BLUE, alpha=0.06)
    ax3.text(-0.14, 1.04, '(c)', transform=ax3.transAxes, weight='bold',
             ha='left', va='bottom', clip_on=False)
    for i, start_sample in enumerate(starts):
        y = nseg - i - 1
        left = start_sample / fs
        width = nperseg / fs
        patch = Rectangle((left, y - 0.24), width, 0.48,
                          facecolor='#DCE8F2', edgecolor=BLACK, lw=0.65)
        ax3.add_patch(patch)
        ax3.text(left + width * 0.5, y, f'Hann {i+1}',
                 ha='center', va='center', fontsize=6.8, color=BLACK)
    ax3.text(0.99, 1.04, f'{nperseg} samples · {overlap_percent:.0f}% overlap',
             transform=ax3.transAxes, fontsize=6.7, ha='right', va='bottom', clip_on=False)

    # No figure title; IEEE caption is outside the image in LaTeX.
    if demo:
        fig.text(0.975, 0.017, 'SYNTHETIC DEMO — not for publication',
                 ha='right', va='bottom', color='#A32020', fontsize=5.8)

    outdir.mkdir(parents=True, exist_ok=True)
    basename = '50_emg_windowing_aligned_ieee_en' + ('_DEMO' if demo else '')
    pdf_path = outdir / f'{basename}.pdf'
    png_path = outdir / f'{basename}.png'
    txt_path = outdir / f'{basename}_audit.txt'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.045)
    fig.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0.045)
    plt.close(fig)

    lines = [
        f'Source: {source}',
        f'Region: {region}, local channel: {channel}, global column: {channel + (64 if region == "FDS" else 0)} (1-based)',
        f'Sampling frequency: {fs:.2f} Hz',
        f'Analysis time: {analysis_time:.2f} s',
        f'Analysis sample index (0-based): {k}',
        f'RMS index interval [start, stop): [{rms_start}, {rms_end})',
        f'RMS points: {rms_end-rms_start}, duration: {rms_duration*1000:.3f} ms',
        f'RMS numerical value: {rms_value:.6g} µV',
        f'Spectral index interval [start, stop): [{spectral_start}, {spectral_end})',
        f'Spectral duration: {spectral_duration:.6f} s',
        f'Hann nperseg: {nperseg}, overlap: {noverlap}, stride: {stride}',
        f'Full Hann segments inside spectral window: {nseg}',
        f'Hann start indices (0-based): {starts}',
        'Feature extraction cadence: 250 ms',
        'Temporal alignment: spectral feature at cache index i+4, 1.0 s ahead of RMS reference.',
        'Spectral support shown: approximately [t-1s, t+1s].',
        'IMPORTANT: aligned feature uses future EMG, OFFLINE only.',
        'Warning: verify the local spectral-feature code uses the same Hann nperseg/overlap.',
    ]
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'Created: {pdf_path}\nCreated: {png_path}\nCreated: {txt_path}')
    return pdf_path, png_path


def main():
    p = argparse.ArgumentParser(description='One real HD-sEMG channel: centered RMS and temporally ALIGNED spectral window')
    p.add_argument('--emg', help='Path to ONE original 128-channel EMG CSV (optional if config/funcoes indexer works).')
    p.add_argument('--region', choices=['EDC', 'FDS'], default='EDC')
    p.add_argument('--channel', type=int, default=12, help='Channel index 1 to 64 within the region')
    p.add_argument('--time', type=float, default=10.0, help='Feature time in seconds; 7.00, 7.25, ..., 40.00')
    p.add_argument('--nperseg', type=int, default=2048, help='Illustrative ~1-s Hann length; verify actual extraction config')
    p.add_argument('--overlap', type=float, default=50.0, help='Hann overlap as a percentage')
    p.add_argument('--outdir', default='resultados_ia753/figures_ieee_color_data_black_axes_en')
    p.add_argument('--synthetic-demo', action='store_true', help='TEST ONLY: create artificial example labeled DEMO')
    args = p.parse_args()
    if not 1 <= args.channel <= 64:
        p.error('--channel must be between 1 and 64')
    if not (7.0 <= args.time <= 40.0):
        p.error('--time must lie between 7 and 40 s')
    if abs((args.time - 7)/0.25 - round((args.time - 7)/0.25)) > 1e-6:
        p.error('--time must be 7.00, 7.25, ..., 40.00 s')
    if not (0 <= args.overlap < 100):
        p.error('--overlap must be between 0 and 100 percent (exclusive)')
    if args.synthetic_demo:
        # Deterministic sample, clearly labeled as synthetic and NOT suitable for the paper.
        rng = np.random.default_rng(27)
        n = int(round(12.0*FS))
        t = np.arange(n)/FS
        x = (45+15*np.sin(2*np.pi*0.5*t)**2) * (
            0.55*np.sin(2*np.pi*70*t) + 0.25*np.sin(2*np.pi*115*t)
            + 0.28*rng.standard_normal(n)
        )
        source = Path('SYNTHETIC_DEMO')
    else:
        source = Path(args.emg) if args.emg else find_example_emg()
        if not source.is_file():
            p.error(f'EMG file does not exist: {source}')
        column = args.channel - 1 + (64 if args.region == 'FDS' else 0)
        print('Reading one real EMG channel from:', source, flush=True)
        x = read_emg_channel(source, column)
    draw_figure(x, analysis_time=args.time, region=args.region, channel=args.channel,
                outdir=Path(args.outdir), nperseg=args.nperseg, overlap_percent=args.overlap,
                demo=args.synthetic_demo, source=source)


if __name__ == '__main__':
    main()

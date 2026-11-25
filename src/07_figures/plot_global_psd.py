# -*- coding: utf-8 -*-
"""Global HD-sEMG PSD (Welch), English IEEE layout.

Fixes NumPy 2.x integration, PSD storage, and tests full storage before full processing.
Requires existing config.py and features.py in the same PyCharm project.

Run: python plot_global_psd.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

import sys

FEATURES_DIR = Path(__file__).resolve().parents[1] / "02_features"
sys.path.insert(0, str(FEATURES_DIR))

import config as C
from features import indexar_gravacoes, ler_exclusoes, ler_csv_numerico

BLUE = "#2C5E91"
ORANGE = "#D97A1B"
BLACK = "#171717"
GRID = "#E0E0E0"

# Known exclusions described in the IA753 manuscript.
# They may already be absent from the indexer; in that case this changes nothing.
KNOWN_EXCLUSIONS = {"Sub017_3_05_450_1", "Sub010_7_05_450_2"}


def compute_recording_psd(emg: np.ndarray):
    """
    128 columns, EMG in microvolts, EDC channels 0:64 and FDS 64:128.
    Welch uses ~1-s Hann segments with 50% overlap, 10-500 Hz range.
    """
    fs = float(getattr(C, "FS_EMG", 2052.52))
    t0 = float(getattr(C, "INTERVALO_INICIO", 5.0))
    t1 = float(getattr(C, "INTERVALO_FIM", 40.0))
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))

    emg = np.asarray(emg)
    if emg.ndim != 2 or emg.shape[1] != 128:
        raise ValueError(f"Expected EMG shape (samples, 128); got {emg.shape}")
    if i0 < 0 or i1 > len(emg) or i1 <= i0:
        raise ValueError(f"Invalid interval [{t0}, {t1}] s for {len(emg)} samples")

    segment = np.asarray(emg[i0:i1, :], dtype=np.float64)
    if not np.all(np.isfinite(segment)):
        raise ValueError("EMG contains NaN/inf within the selected interval")

    nperseg = int(round(fs))
    frequencies, pxx = welch(
        segment,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    selected = (frequencies >= 10.0) & (frequencies <= 500.0)
    f = frequencies[selected]
    pxx = pxx[selected, :]

    # IMPORTANT: define these arrays BEFORE computing the integrals.
    edc = np.mean(pxx[:, :64], axis=1)
    fds = np.mean(pxx[:, 64:128], axis=1)

    # NumPy >=2.0: `trapezoid` replaces removed `trapz`.
    edc_area = float(np.trapezoid(edc, x=f))
    fds_area = float(np.trapezoid(fds, x=f))
    if not np.isfinite(edc_area) or edc_area <= 0:
        raise ValueError(f"EDC integrated PSD is invalid: {edc_area}")
    if not np.isfinite(fds_area) or fds_area <= 0:
        raise ValueError(f"FDS integrated PSD is invalid: {fds_area}")

    return f, edc, fds, edc / edc_area, fds / fds_area


def mean_curves(curves):
    if not curves:
        raise ValueError("No curves available for aggregation")
    return np.mean(np.stack(curves), axis=0)


def aggregate_psd():
    recordings, issues = indexar_gravacoes()
    excluded = ler_exclusoes()
    excluded = {str(x).replace(".csv", "").strip() for x in excluded}
    excluded |= KNOWN_EXCLUSIONS
    selected = []
    for g in recordings:
        if int(g.tarefa) not in range(1, 9):
            continue
        # Prefixes such as Sub001__Sub001_1_05_450_0 are supported.
        base = Path(g.emg).stem
        normalized_base = base.split("__")[-1]
        if base in excluded or normalized_base in excluded or str(g.id).split("__")[-1] in excluded:
            continue
        selected.append(g)

    print(f"Indexed recordings: {len(recordings)}; selected after exclusions: {len(selected)}")
    print(f"Indexer notes: {len(issues)}")
    if not selected:
        raise RuntimeError("No matching EMG recordings were found")

    # Process first recording OUTSIDE the try/except loop: fail immediately
    # with a precise traceback instead of skipping 1007 recordings silently.
    first = selected[0]
    print(f"Preflight: {first.id}")
    first_emg = ler_csv_numerico(first.emg, 128)
    first_psd = compute_recording_psd(first_emg)
    print(f"Preflight OK: {len(first_psd[0])} frequency bins, "
          f"EDC integral = {np.trapezoid(first_psd[1], first_psd[0]):.4g}")

    # Plain list per (participant/task/cadence, muscle, representation).
    # The previous nested defaultdict had no .append() and silently skipped every recording.
    condition = defaultdict(list)
    frequency_axis = first_psd[0]
    failures = []

    def store_curve(recording, data):
        f, edc, fds, edc_normalized, fds_normalized = data
        if len(f) != len(frequency_axis) or not np.allclose(f, frequency_axis):
            raise ValueError("Welch frequency grid changed between recordings")
        cadence = float(recording.frequencia)
        if cadence not in (0.5, 0.75):
            raise ValueError(f"Unexpected cadence: {cadence}")
        key = (str(recording.sujeito), int(recording.tarefa), cadence)
        condition[(key, "EDC", "absolute")].append(edc)
        condition[(key, "FDS", "absolute")].append(fds)
        condition[(key, "EDC", "normalized")].append(edc_normalized)
        condition[(key, "FDS", "normalized")].append(fds_normalized)

    # Critical: test the ENTIRE storage pathway before the lengthy 1,006-file pass.
    store_curve(first, first_psd)
    if len(condition) != 4 or not all(len(v) == 1 for v in condition.values()):
        raise RuntimeError("Preflight failed while saving the first four PSD curves")
    print("Preflight storage OK: four PSD curves registered. Starting full dataset...")

    for i, g in enumerate(selected[1:], start=2):
        try:
            data = compute_recording_psd(ler_csv_numerico(g.emg, 128))
            store_curve(g, data)
        except Exception as exc:
            failures.append((str(g.id), str(exc)))
            if len(failures) == 1:
                print(f"WARNING, first skipped recording: {g.id}: {exc}")
            # Repeated identical failures indicate a programming error, not bad recordings.
            if len(failures) >= 3 and len({x[1] for x in failures[-3:]}) == 1:
                raise RuntimeError(
                    "Three recordings failed for the same reason; stopping early "
                    "to avoid wasting processing time. Last error: " + str(exc)
                ) from exc
        if i % 50 == 0 or i == len(selected):
            print(f"Processed {i}/{len(selected)} recordings")

    print(f"Skipped recordings: {len(failures)}")
    if failures:
        for name, error in failures[:8]:
            print("  ", name, "->", error)
    if not condition:
        raise RuntimeError("No valid recordings remained after preprocessing")

    # Repetitions within subject-task-cadence, then tasks within subject-cadence,
    # then participants: each participant contributes equally to the grand mean.
    by_subject = defaultdict(list)
    for (sub_task_cad, region, kind), curves in condition.items():
        subject, task, cadence = sub_task_cad
        by_subject[(subject, cadence, region, kind)].append(mean_curves(curves))

    by_group = defaultdict(list)
    for (subject, cadence, region, kind), task_curves in by_subject.items():
        by_group[(cadence, region, kind)].append(mean_curves(task_curves))

    result = {}
    for key, subject_curves in by_group.items():
        a = np.stack(subject_curves)
        result[(key[1], key[0], key[2])] = {
            "f": frequency_axis,
            "mean": np.mean(a, axis=0),
            "std": np.std(a, axis=0, ddof=1) if len(a) > 1 else np.zeros_like(frequency_axis),
            "n_subjects": len(a),
        }
    for region in ("EDC", "FDS"):
        for cadence in (0.5, 0.75):
            for kind in ("absolute", "normalized"):
                if (region, cadence, kind) not in result:
                    raise RuntimeError(f"Missing result {(region, cadence, kind)}")

    if any(x["n_subjects"] != 21 for x in result.values()):
        print("WARNING: Some curves do not have 21 participants. "
              "Check exclusions before using this figure in the manuscript.")

    return result, failures


def plot(result, output: Path, dpi: int):
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.labelsize": 9, "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")
    for row, region in enumerate(("EDC", "FDS")):
        for col, kind in enumerate(("absolute", "normalized")):
            ax = axes[row, col]
            for cadence, color in ((0.5, BLUE), (0.75, ORANGE)):
                d = result[(region, cadence, kind)]
                label = f"{cadence:.2f} Hz (n={d['n_subjects']})"
                ax.plot(d["f"], d["mean"], color=color, lw=1.35, label=label)
                if kind == "normalized":
                    ax.fill_between(
                        d["f"], np.maximum(0, d["mean"] - d["std"]),
                        d["mean"] + d["std"], color=color, alpha=0.15, lw=0
                    )
            if kind == "absolute":
                ax.set_yscale("log")
                ax.set_ylabel("PSD (µV²/Hz)")
            else:
                ax.set_ylim(bottom=0)
                ax.set_ylabel("Normalized PSD (1/Hz)")
            ax.set_xlim(0, 500)
            ax.set_xlabel("EMG frequency (Hz)")
            ax.text(0.025, 0.94, f"({chr(97 + row*2 + col)}) {region}",
                    transform=ax.transAxes, va="top", ha="left", color=BLACK,
                    fontsize=8, fontweight="bold")
            ax.legend(loc="upper right", frameon=True, facecolor="white",
                      edgecolor="#CCCCCC", fontsize=7)
            ax.grid(True, color=GRID, linewidth=0.45, alpha=0.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color(BLACK)

    for ext in ("pdf", "png"):
        path = output / f"42_psd_global_publication_consistent_en.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None)
        print(f"Created: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="resultados_ia753/figures_ieee_final_en")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result, failures = aggregate_psd()
    rows = []
    for (region, cadence, kind), d in result.items():
        for f, mean, std in zip(d["f"], d["mean"], d["std"]):
            rows.append({
                "region": region, "frequency_hz": cadence, "kind": kind,
                "emg_frequency_hz": f, "mean_psd": mean, "std_psd": std,
                "n_subjects": d["n_subjects"],
            })
    pd.DataFrame(rows).to_csv(outdir / "42_psd_global_summary_en.csv", index=False)
    if failures:
        pd.DataFrame(failures, columns=["recording", "error"]).to_csv(
            outdir / "42_psd_skipped_recordings.csv", index=False
        )
    plot(result, outdir, args.dpi)
    print("Global PSD figure and CSV generated successfully.")


if __name__ == "__main__":
    main()

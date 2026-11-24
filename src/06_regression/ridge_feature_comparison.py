

"""
Comparação de características para regressão Ridge multissaída.
Avalia seis configurações de características HD-sEMG para
estimar os ângulos dos cinco dedos, utilizando modelos
específicos por participante e cadência.

As características espectrais MNF e MDF são alinhadas
temporalmente ao RMS por um deslocamento de quatro passos
de 250 ms.

O protocolo utiliza:
- Oito tarefas por participante, com a repetição 0.
- Separação temporal entre treinamento e teste.
- Intervalo de proteção de 2,25 s.
- StandardScaler ajustado apenas no treinamento.
- Regressão Ridge com alpha = 10.
- Avaliação de MAE, RMSE e R² ponderado.

Inclui um controle temporal com RMS suavizado, calculado
a partir das características previamente extraídas.

"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATTERN = re.compile(r"^(?:Sub\d{3}__)?(Sub\d{3})_([1-8])_(05|075)_450_(\d+)$")
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Little")
SHIFT = 4
MODELS = (
    "RMS",
    "RMS+MNF",
    "RMS+MDF",
    "RMS+MNF+MDF",
    "MNF+MDF",
    "RMS+RMS_context",
)
PRIMARY_MODELS = ("RMS+MNF", "RMS+MDF", "RMS+MNF+MDF", "MNF+MDF")
SECONDARY_PAIRS = (
    ("RMS+MNF", "RMS+RMS_context"),
    ("RMS+MNF+MDF", "RMS+RMS_context"),
    ("RMS+RMS_context", "RMS"),
)


def load_cache(path: Path) -> dict:
    fields = ("rms", "mnf", "mdf", "angulos", "tempo_s")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in fields if key not in data.files]
        if missing:
            raise ValueError(f"Missing NPZ fields {missing} in {path.name}")
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in fields}
    times = arrays["tempo_s"]
    n = len(times)
    if times.ndim != 1 or n < 45:
        raise ValueError(f"Invalid/short time axis: {path.name}")
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), .25, atol=.002, rtol=0):
        raise ValueError(f"Expected finite 250-ms time grid: {path.name}")
    if arrays["angulos"].shape != (n, 5):
        raise ValueError(f"Expected {n}x5 angles, got {arrays['angulos'].shape}")
    for field in ("rms", "mnf", "mdf"):
        if arrays[field].shape != (n, 128):
            raise ValueError(f"Expected {n}x128 for {field}, got {arrays[field].shape}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Non-finite feature or target in {path.name}")
    return arrays


def load_groups(cache_dir: Path, cadences: set[float], max_subjects: int | None):
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Missing cache directory: {cache_dir}")
    index = {}
    for path in sorted(cache_dir.glob("*.npz")):
        match = PATTERN.fullmatch(path.stem)
        if not match or match.group(4) != "0":
            continue
        subject, task, cadence_code, _ = match.groups()
        cadence = .5 if cadence_code == "05" else .75
        if cadence not in cadences:
            continue
        key = (cadence, subject, int(task))
        if key in index:
            raise ValueError(f"Duplicate subject/cadence/task cache: {key}")
        index[key] = path
    subjects_by_cadence = {
        cadence: sorted({subject for (cad, subject, _) in index if cad == cadence})
        for cadence in cadences
    }
    selected = {
        cadence: set(subjects[:max_subjects] if max_subjects else subjects)
        for cadence, subjects in subjects_by_cadence.items()
    }
    groups = defaultdict(list)
    for (cadence, subject, task), path in sorted(index.items()):
        if subject not in selected[cadence]:
            continue
        arrays = load_cache(path)
        groups[(cadence, subject)].append(dict(task=task, file=path.name, **arrays))
    for (cadence, subject), records in groups.items():
        tasks = sorted(r["task"] for r in records)
        if tasks != list(range(1, 9)):
            raise ValueError(f"Incomplete tasks for {subject} at {cadence} Hz: {tasks}")
    for cadence in cadences:
        if not any(cad == cadence for cad, _ in groups):
            raise ValueError(f"No eligible complete subjects at {cadence} Hz")
    return groups


def aligned_features(record: dict) -> dict:
    """Construct *all* six models on the exact rows used in Experiment 47."""
    n = len(record["tempo_s"])
    ids = np.arange(SHIFT, n - SHIFT, dtype=int)
    rms = record["rms"]
    mnf = record["mnf"][ids + SHIFT]
    mdf = record["mdf"][ids + SHIFT]
    rms_now = rms[ids]
    sq = np.square(rms)
    cumulative = np.vstack([np.zeros((1, 128)), np.cumsum(sq, axis=0)])
    # Nine 250 ms centers from t-1 to t+1 seconds.
    context = np.sqrt(np.maximum(
        (cumulative[ids + SHIFT + 1] - cumulative[ids - SHIFT]) / (2 * SHIFT + 1),
        0.,
    ))
    matrices = {
        "RMS": rms_now,
        "RMS+MNF": np.hstack((rms_now, mnf)),
        "RMS+MDF": np.hstack((rms_now, mdf)),
        "RMS+MNF+MDF": np.hstack((rms_now, mnf, mdf)),
        "MNF+MDF": np.hstack((mnf, mdf)),
        "RMS+RMS_context": np.hstack((rms_now, context)),
    }
    for name, x in matrices.items():
        if not np.isfinite(x).all():
            raise ValueError(f"Invalid assembled features: {record['file']} / {name}")
    return {"times": record["tempo_s"][ids], "targets": record["angulos"][ids],
            "models": matrices, "ids": ids}


def split_and_stack(records: list[dict], guard: float):
    train_x, test_x = {m: [] for m in MODELS}, {m: [] for m in MODELS}
    train_y, test_y, audit = [], [], []
    for record in sorted(records, key=lambda r: r["task"]):
        feat = aligned_features(record)
        midpoint = (float(record["tempo_s"][0]) + float(record["tempo_s"][-1])) / 2
        train_ids = np.flatnonzero(feat["times"] <= midpoint - guard)
        test_ids = np.flatnonzero(feat["times"] >= midpoint + guard)
        if min(len(train_ids), len(test_ids)) < 10:
            raise ValueError(f"Too few train/test windows for {record['file']}")
        # Conservative feature raw support (MNF aligned +/-1s, context +/-1.15s)
        last_raw_train = float(feat["times"][train_ids[-1]] + 1.151)
        first_raw_test = float(feat["times"][test_ids[0]] - 1.151)
        if last_raw_train >= first_raw_test:
            raise AssertionError(f"Potential train/test raw-support overlap in {record['file']}")
        for model in MODELS:
            train_x[model].append(feat["models"][model][train_ids])
            test_x[model].append(feat["models"][model][test_ids])
        train_y.append(feat["targets"][train_ids])
        test_y.append(feat["targets"][test_ids])
        audit.append({
            "recording": record["file"], "task": record["task"],
            "n_training": len(train_ids), "n_testing": len(test_ids),
            "last_training_time_s": float(feat["times"][train_ids[-1]]),
            "first_testing_time_s": float(feat["times"][test_ids[0]]),
            "conservative_raw_support_gap_s": first_raw_test - last_raw_train,
        })
    xtr = {model: np.vstack(parts) for model, parts in train_x.items()}
    xte = {model: np.vstack(parts) for model, parts in test_x.items()}
    return xtr, xte, np.vstack(train_y), np.vstack(test_y), audit


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    variance = np.var(y_true, axis=0)
    if float(variance.sum()) <= 1e-12:
        raise ValueError("Zero variance of all angular targets")
    individual_r2 = np.array([
        r2_score(y_true[:, j], y_pred[:, j]) if variance[j] > 1e-12 else np.nan
        for j in range(5)
    ])
    r2_weighted = float(np.nansum((variance / variance.sum()) * individual_r2))
    metrics = dict(
        MAE_deg=float(mean_absolute_error(y_true, y_pred)),
        RMSE_deg=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        R2_vw=r2_weighted,
    )
    finger_rows = [
        dict(
            finger=FINGER_NAMES[j],
            MAE_deg=float(mean_absolute_error(y_true[:, j], y_pred[:, j])),
            RMSE_deg=float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))),
            R2=float(individual_r2[j]) if np.isfinite(individual_r2[j]) else np.nan,
        ) for j in range(5)
    ]
    return metrics, finger_rows


def holm_correction(values):
    p = np.asarray(values, dtype=float)
    output = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.
    for rank, position in enumerate(order):
        running = max(running, min(1., (len(order) - rank) * float(p[position])))
        output[position] = running
    return output


def pairwise_tests(detail: pd.DataFrame):
    """Holm across 8 primary comparisons and separately 6 secondary controls."""
    rows = []
    for cadence in sorted(detail["cadence_hz"].unique()):
        pivot = detail[np.isclose(detail["cadence_hz"], cadence)].pivot(
            index="subject", columns="model", values="MAE_deg"
        )
        pairs = [(m, "RMS", "primary_8") for m in PRIMARY_MODELS]
        pairs += [(a, b, "secondary_6") for a, b in SECONDARY_PAIRS]
        for model, reference, family in pairs:
            matched = pivot[[model, reference]].dropna()
            difference = (matched[model] - matched[reference]).to_numpy()
            if len(difference) < 5:
                raw_p = np.nan
            elif np.all(difference == 0):
                raw_p = 1.
            else:
                raw_p = float(wilcoxon(difference, alternative="two-sided").pvalue)
            rows.append(dict(
                family=family, cadence_hz=cadence, model=model, reference=reference,
                n_subjects=len(matched), mean_delta_MAE_deg=float(difference.mean()),
                median_delta_MAE_deg=float(np.median(difference)),
                n_better=int((difference < 0).sum()), p_wilcoxon=raw_p,
            ))
    comparisons = pd.DataFrame(rows)
    comparisons["p_holm"] = np.nan
    for family in comparisons["family"].unique():
        mask = comparisons["family"] == family
        comparisons.loc[mask, "p_holm"] = holm_correction(
            comparisons.loc[mask, "p_wilcoxon"].to_numpy()
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Diretório dos resultados utilizados pela extração de características.
    DIRETORIO_RESULTADOS = (
            Path(__file__).resolve().parents[1]
            / "02_features"
            / "resultados_ia753"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DIRETORIO_RESULTADOS,
        help="Diretório que contém o cache features_welch_centrado"
    )

    parser.add_argument("--cadences", type=float, nargs="+", default=[.5, .75],
                        choices=[.5, .75])
    parser.add_argument("--alpha", type=float, default=10.)
    parser.add_argument("--guard", type=float, default=2.25)
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Smoke test only; e.g. --max-subjects 2")
    args = parser.parse_args()
    if args.alpha <= 0 or args.guard < 2.25:
        parser.error("--alpha must be >0; --guard must be >=2.25 s")
    if args.max_subjects is not None and args.max_subjects < 1:
        parser.error("--max-subjects must be a positive integer")
    base = args.base
    out = base / "48_ridge_full_aligned_ablation"
    out.mkdir(parents=True, exist_ok=True)
    groups = load_groups(base / "features_welch_centrado", set(args.cadences), args.max_subjects)
    if args.max_subjects is None:
        for cadence in sorted(set(args.cadences)):
            count = sum(c == cadence for c, _ in groups)
            if count != 21:
                raise RuntimeError(f"Expected exactly 21 complete participants at {cadence} Hz; found {count}. "
                                   "Check NPZ cache integrity or use --max-subjects only for a short test.")
    print(f"Found {len(groups)} complete subject x cadence groups", flush=True)

    rows, per_finger_rows, audit_rows = [], [], []
    for cadence in sorted(set(args.cadences)):
        subjects = sorted(subject for cad, subject in groups if cad == cadence)
        print(f"\nCadence {cadence:.2f} Hz: {len(subjects)} subjects", flush=True)
        for count, subject in enumerate(subjects, 1):
            xtr, xte, ytr, yte, audit = split_and_stack(groups[(cadence, subject)], args.guard)
            y_baseline = np.broadcast_to(ytr.mean(axis=0), yte.shape)
            baseline_metrics, _ = evaluate(yte, y_baseline)
            for rec in audit:
                audit_rows.append(dict(cadence_hz=cadence, subject=subject, **rec))
            for model in MODELS:
                pipe = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
                pipe.fit(xtr[model], ytr)
                prediction = pipe.predict(xte[model])
                metric, finger_metrics = evaluate(yte, prediction)
                common = dict(cadence_hz=cadence, subject=subject, model=model)
                rows.append(dict(**common, n_features=xtr[model].shape[1],
                                 n_train=len(ytr), n_test=len(yte), alpha=args.alpha,
                                 baseline_MAE_deg=baseline_metrics["MAE_deg"], **metric))
                per_finger_rows.extend(dict(**common, **record) for record in finger_metrics)
            print(f"  {count}/{len(subjects)} {subject}: {len(ytr)} train, "
                  f"{len(yte)} test; {len(MODELS)} fits OK", flush=True)

    detail = pd.DataFrame(rows)
    by_finger = pd.DataFrame(per_finger_rows)
    summary = (detail.groupby(["cadence_hz", "model"], as_index=False)
               .agg(n_subjects=("subject", "nunique"),
                    MAE_mean=("MAE_deg", "mean"),
                    MAE_median=("MAE_deg", "median"),
                    MAE_std=("MAE_deg", "std"),
                    RMSE_mean=("RMSE_deg", "mean"),
                    R2_vw_mean=("R2_vw", "mean"),
                    R2_vw_median=("R2_vw", "median"),
                    baseline_MAE_mean=("baseline_MAE_deg", "mean")))
    finger_summary = (by_finger.groupby(["cadence_hz", "model", "finger"], as_index=False)
                      .agg(n_subjects=("subject", "nunique"),
                           MAE_mean=("MAE_deg", "mean"),
                           R2_mean=("R2", "mean")))
    comparisons = pairwise_tests(detail)

    # Cross-check the three shared models against the earlier real Experiment 47.
    # This check runs ONLY when alpha/guard and participant sets match.
    ref_path = base / '47_ridge_controles_temporais' / '47_ridge_por_participante.csv'
    if ref_path.exists() and args.alpha == 10.0 and args.guard == 2.25:
        reference = pd.read_csv(ref_path)
        mapping = {'RMS': 'RMS', 'RMS+MNF_aligned': 'RMS+MNF',
                   'RMS+RMS_context': 'RMS+RMS_context'}
        comparable = True
        for old_model, new_model in mapping.items():
            older = reference[reference['model'] == old_model]
            newer = detail[detail['model'] == new_model]
            ref_keys = set(zip(older['cadence_hz'], older['subject']))
            new_keys = set(zip(newer['cadence_hz'], newer['subject']))
            if ref_keys != new_keys:
                comparable = False
                break
        if comparable:
            largest_difference = 0.
            for old_model, new_model in mapping.items():
                older = reference[reference['model'] == old_model][
                    ['cadence_hz', 'subject', 'MAE_deg', 'R2_vw']]
                newer = detail[detail['model'] == new_model][
                    ['cadence_hz', 'subject', 'MAE_deg', 'R2_vw']]
                merged = older.merge(newer, on=['cadence_hz', 'subject'], suffixes=('_47','_48'))
                for metric in ('MAE_deg', 'R2_vw'):
                    difference = float(np.max(np.abs(merged[metric+'_47'] - merged[metric+'_48'])))
                    largest_difference = max(largest_difference, difference)
            if largest_difference > 1e-7:
                raise AssertionError(f'Experiment 47/48 mismatch: {largest_difference:.6g}. '
                                     'Do not publish without reconciling the discrepancy.')
            print(f'Experiment 47 cross-check PASSED: largest MAE/R2 difference {largest_difference:.3g}',
                  flush=True)
        else:
            print('Experiment 47 comparison skipped (different subject subsets).', flush=True)

    # Extra consistency checks: complete 6-model/5-finger grid per participant.
    model_counts = detail.groupby(["cadence_hz", "subject"])["model"].nunique()
    if not (model_counts == len(MODELS)).all():
        raise AssertionError("Some participants lack aligned model results")
    finger_counts = by_finger.groupby(["cadence_hz", "subject", "model"])["finger"].nunique()
    if not (finger_counts == 5).all():
        raise AssertionError("Some models lack per-finger results")

    detail.to_csv(out / "48_ridge_aligned_by_subject.csv", index=False)
    by_finger.to_csv(out / "48_ridge_aligned_by_finger.csv", index=False)
    summary.to_csv(out / "48_ridge_aligned_summary.csv", index=False)
    finger_summary.to_csv(out / "48_ridge_aligned_finger_summary.csv", index=False)
    comparisons.to_csv(out / "48_ridge_aligned_paired_tests.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "48_temporal_split_audit.csv", index=False)

    print("\n== ALIGNED FEATURE ABLATION ==", flush=True)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print("\n== PAIRED WILCOXON + HOLM ==", flush=True)
    print(comparisons.to_string(index=False, float_format=lambda v: f"{v:.5g}"), flush=True)
    print("\nFiles created in:", out.resolve(), flush=True)
    print("IMPORTANT: aligned spectral features use future EMG information; offline only.")
    print("IMPORTANT: RMS_context is a cache-based proxy, NOT 2-s raw-signal RMS.")
    print("IMPORTANT: all models use common trimmed rows; not identical to Exp.26.")


if __name__ == "__main__":
    main()

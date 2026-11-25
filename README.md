# HD-sEMG Cadence Analysis

**Temporal and Spectral Analysis of HD-sEMG During Hand Movements at Different Cadences**

Research code accompanying a study developed for **IA753 – Biological Signal Analysis** at the **University of Campinas (UNICAMP), Brazil**.

This repository contains Python scripts for the secondary analysis of a publicly available high-density surface electromyography (HD-sEMG) and finger-kinematics dataset. The original recordings were acquired and released by the dataset authors; the present work focuses on signal processing, statistical analysis, and reproducibility.

## Study Overview

The study investigates how prescribed movement cadences of **0.50 Hz** and **0.75 Hz** relate to HD-sEMG amplitude, spectral characteristics, and finger kinematics.

| Parameter | Description |
|---|---|
| Participants | 21 |
| Motor tasks | 8 |
| Cadences | 0.50 and 0.75 Hz |
| HD-sEMG channels | 128 (64 EDC + 64 FDS) |
| EMG sampling frequency | 2052.52 Hz |
| Kinematic sampling frequency | 100 Hz |
| Analyzed recordings | 1,006 |
| Feature time grid | 7–40 s, 250 ms step |
| Spectral band | 10–500 Hz |

The electrode arrays were positioned over the **extensor digitorum communis (EDC)** and **flexor digitorum superficialis (FDS)** regions.

The eight motor tasks include isolated finger movements, finger–thumb pinches, tripod pinch, and five-finger grasp.

## Main Analyses

### Temporal and Spectral Features

- Root Mean Square (RMS) and Average Rectified Value (ARV), using centered 617-sample windows (approximately 300.6 ms).
- Mean Frequency (MNF) and Median Frequency (MDF), using Welch-based spectral estimation.
- Global absolute and normalized Power Spectral Density (PSD) over 10–500 Hz.
- Interchannel RMS coefficient of variation (CV) across each 64-channel region.

### Statistical Analysis

- Participant-level aggregation across recordings and repetitions.
- Paired Wilcoxon signed-rank tests with Holm correction.
- Task-wise percentage change in mean RMS, with pointwise 95% confidence intervals from 10,000 paired participant-bootstrap resamples.
- Exploratory angular-excursion comparisons based on the P95–P5 range of reference-digit angles.

### EMG–Kinematic Relationships

- RMS–angle cross-correlation at lags from −1.0 to +1.0 s in 0.25 s steps.
- Zero-lag correlation heatmaps for eight tasks and two cadences.
- Temporal alignment of RMS and spectral features for offline analyses.
- Participant-specific, multi-output Ridge regression for estimating five finger-joint angles.
- Comparison of RMS, spectral feature combinations, and an RMS-based temporal-context control.

## Repository Structure

```text
HD-sEMG-Cadence-Analysis/
├── src/
│   ├── 01_dataset/
│   │   └── audit_dataset.py
│   ├── 02_features/
│   │   ├── config.py
│   │   ├── features.py
│   │   └── extract_features.py
│   ├── 03_spectral/
│   │   └── global_psd.py
│   ├── 04_statistics/
│   │   └── global_analysis.py
│   ├── 05_kinematics/
│   │   └── cross_correlation.py
│   ├── 06_regression/
│   │   └── ridge_feature_comparison.py
│   ├── 07_figures/
│   │   ├── plot_emg_windowing.py
│   │   ├── plot_rms_change.py
│   │   ├── plot_global_psd.py
│   │   ├── plot_rms_cv.py
│   │   ├── plot_correlation_heatmap.py
│   │   ├── plot_cross_correlation_lags.py
│   │   └── plot_ridge_results.py
│   └── 08_validation/
│       ├── validate_angular_excursion.py
│       └── validate_final_results.py
├── data/
├── results/
├── paper/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above describes the intended organization. Some local data files, intermediate results, and output directories are not distributed with the source code.

## Installation

Python 3.12 was used for local development.

```bash
git clone <REPOSITORY_URL>
cd HD-sEMG-Cadence-Analysis
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The core dependencies are NumPy, pandas, SciPy, Matplotlib, and scikit-learn.

## Dataset

The analysis uses a publicly available dataset associated with the HD-sEMG platform developed by Molinari and collaborators and released by Avilés-Carrillo and collaborators.

The dataset contains multichannel HD-sEMG recordings, hand kinematics, and experimental task information.

**The original dataset was not collected as part of this study and is not redistributed in this repository.**

Obtain the recordings from the original dataset source and consult the accompanying manuscript for full bibliographic information, experimental details, and attribution.

## Manuscript Figures

| Figure | Description | Generator |
|---|---|---|
| 1 | Forearm anatomy | Adapted source illustration |
| 2 | Eight hand-movement tasks | Adapted dataset illustration |
| 3 | Temporal alignment of RMS and Welch windows | `plot_emg_windowing.py` |
| 4 | Mean RMS percentage change with bootstrap 95% CI | `plot_rms_change.py` |
| 5 | Absolute and normalized global PSD | `plot_global_psd.py` |
| 6 | Interchannel RMS CV across tasks | `plot_rms_cv.py` |
| 7 | Zero-lag RMS–angle correlation heatmap | `plot_correlation_heatmap.py` |
| 8 | Lag-dependent RMS–angle correlation | `plot_cross_correlation_lags.py` |
| 9 | Participant-level Ridge MAE distributions | `plot_ridge_results.py` |

Figure scripts use raw recordings or intermediate analysis outputs, depending on the figure.

### Example: RMS Cadence Comparison

From the repository root:

```powershell
python .\src\07_figures\plot_rms_change.py `
  --csv "PATH_TO_RESULTS\18_analise_por_tarefa_centrado\18_por_sujeito_condicao.csv" `
  --reference "PATH_TO_RESULTS\18_analise_por_tarefa_centrado\18_resultados_wilcoxon_holm.csv" `
  --saida "PATH_TO_OUTPUT"
```

### Example: Zero-Lag Correlation Heatmap

```powershell
python .\src\07_figures\plot_correlation_heatmap.py `
  --csv "PATH_TO_RESULTS\20_final_centrado\20_correlacao_tau_zero.csv" `
  --saida "PATH_TO_OUTPUT"
```

Other scripts may require intermediate NPZ/CSV files and dataset-specific configuration. A fully automated end-to-end workflow from a clean checkout has not yet been verified.

## Validation

The repository includes two complementary validation scripts.

### Angular Excursion

`validate_angular_excursion.py` computes the robust angular excursion (P95–P5) for the reference digit of each task.

The local audit processed **1,006 recordings** and compared **21 paired participants per task**. After Holm correction, differences between cadences remained significant for T1, T5, T7, and T8.

### Final Statistical Checks

`validate_final_results.py` checks the angular-excursion summary, exploratory task-level Spearman correlations, and selected Ridge metrics.

In the local participant-level Ridge audit, mean variance-weighted R² for RMS+MNF was:

| Cadence | Mean R² |
|---|---:|
| 0.50 Hz | 0.771127 |
| 0.75 Hz | 0.771150 |

The final validation script was tested in the original analysis directory. Its results-directory paths must be adapted or configured when executing from this reorganized repository.

## Scientific Limitations

- HD-sEMG amplitude is not a direct measurement of muscle force.
- MNF and MDF changes alone cannot establish muscle fatigue.
- Correlation does not establish causation or electromechanical delay.
- Spectral-window alignment includes future signal samples and is therefore offline.
- Ridge models were calibrated and evaluated within each participant.
- Generalization to unseen participants and real-time operation was not evaluated.
- The global PSD comparisons are descriptive; no frequency-by-frequency significance claims are made.

## Attribution and References

The original recordings, experimental protocol, and source illustrations are attributed to their respective authors in the accompanying manuscript:

**Temporal and Spectral Analysis of HD-sEMG During Hand Movements at Different Cadences**.

The present repository documents the secondary analysis and does not claim authorship of the original dataset or acquisition platform.

## Author

**Jeison A. Magallanes Poma**  
Faculty of Electrical and Computer Engineering (FEEC)  
University of Campinas (UNICAMP), Brazil

Research interests: biomedical signal processing, HD-sEMG, machine learning, and digital hardware.

## Project Status

Research scripts and manuscript figure generators have been organized. Local validation of key statistical results has been completed.

Independent end-to-end execution from a clean checkout still requires dataset configuration, intermediate-file dependencies, and path verification.

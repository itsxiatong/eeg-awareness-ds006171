# Reproducibility record

This file records how the reported results were produced, and what has and has not been verified.

## 1. Environments actually used

| Stage | Environment | Key versions |
|---|---|---|
| Preprocessing (`01_preprocess.py`, `01b_make_compact.py`) | Linux aarch64 VM, Python 3.10.12 | MNE 1.12.1, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3 |
| Group analyses (`02`–`07`, notebook) | Linux x86_64, Python 3.13.17 | MNE 1.13.2, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, statsmodels 0.15.0, pingouin 0.7.0 |

Preprocessing ran in a VM limited to ~3 GB RAM and ~3 min per process, which is why `01_preprocess.py` filters channels in groups and saves checkpoints (`derivatives/tmp_sub-XX/`) so that an interrupted run resumes where it stopped.

## 2. What was run, and how often

| Step | Command | Runs | Outputs |
|---|---|---|---|
| Download | `openneuro-py download --dataset ds006171 --include "sub-*/eeg/*task-nocue*"` | once | 35 participants (no `sub-35` in the release) |
| Preprocessing | `python3 code/01_preprocess.py sub-XX` for all 35 participants | final parameters: once per participant (earlier pilot runs on 5 participants used other parameters and were superseded) | `derivatives/sub-XX_task-nocue_epo.fif`, `sub-XX_qc.csv` |
| Downsampled copies | `python3 code/01b_make_compact.py` | once | `derivatives/compact/*_desc-128Hz_epo.fif`, `qc_all.csv` |
| Group analyses | `EEG_awareness_analysis.ipynb`, executed top to bottom in a fresh kernel (runs `02`, `03`, `05`, `06`) | final run 2026-10-24, no errors | all files in `output/` |
| Unrounded cluster p-values | recomputed from `group_data.npz` with the same seed | 2026-10-24 | `p_permutation_unrounded` (empirical permutation p-value, subject to Monte Carlo sampling) and permutation counts in `results_tf_clusters.csv` |
| Audit | `python3 code/07_audit_manuscript.py manuscript.txt` | after every manuscript revision | `output/AUDIT.md` (34/34) |

## 3. What each check establishes

| Level | Covered by | Recomputed from |
|---|---|---|
| Text ↔ result files | `07_audit_manuscript.py` | means, t tests, proportions: per-subject values; cluster p-values, decoding curves, time–frequency arrays: read from saved files |
| Result files ↔ epochs | notebook final run | per-subject ERPs, time–frequency power, decoding, cluster permutation tests, contrast control, blocked CV: all recomputed from the 128-Hz epochs |
| Epochs ↔ raw EEG | `01_preprocess.py` | run once from the raw BIDS files |
| Independent rerun | — | **not yet done**: no one has rerun the full pipeline from raw data in a separate environment |

## 4. Known sources of run-to-run variation
- ICA (FastICA, `random_state=0`) and permutation tests (`seed=0`) are seeded; contrast matching uses seed 1.
- Different MNE / scikit-learn versions can change ICA solutions or filter edge handling slightly, so an independent rerun may differ in the last decimal places of some statistics.
- Rounding of reported p values uses decimal round-half-up (e.g., the unrounded permutation p = 13/2000 = .0065 is reported as .007).

## 5. For an independent rerun
Follow the README from a clean environment, then run `07_audit_manuscript.py` on the manuscript text. Any value marked `NO` in `AUDIT.md` identifies a statistic that differs from the reported one.

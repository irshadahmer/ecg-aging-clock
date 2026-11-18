# Deep-Learning ECG Aging Clock (12-Lead Clinical & Lead-I Smartwatch Distillation)

A reproducible deep-learning framework estimating biological cardiovascular age from 10-second 100 Hz ECG waveforms on the **PTB-XL** dataset (N = 21,373 recordings; patient-grouped splits).

## Architecture & Pipeline
- **Preprocessing (src/preprocess.py):** 0.5-40 Hz 4th-order Butterworth bandpass filter, per-lead z-score normalization, and leak-free patient-grouped splits (Train: 17,090; Val: 2,132; Test: 2,151).
- **Model (src/model.py):** 1D-ResNet with Squeeze-and-Excitation (SE) channel attention and dual average/max temporal pooling.
- **Knowledge Distillation (src/train.py):** 12-Lead clinical teacher distilled into a single-lead (Lead I) smartwatch student using Smooth L1 loss.
- **Statistical & Clinical Audit (src/evaluate.py):** Validation-fitted linear residualization to eliminate age bias, 1,000x patient-clustered bootstrap 95% CIs, subgroup fairness audits, and age/sex-adjusted logistic regression for cardiovascular comorbidities.

## Key Test-Set Results (n = 2,151)
| Model | Raw Val MAE | Bias-Corrected Test MAE (95% CI) | Pearson r | Post-Correction Age-Bias r | MI Adjusted OR | ST/T Adjusted OR |
|---|---|---|---|---|---|---|
| **12-Lead Teacher** | 7.59 yrs | 9.29 (8.95-9.61) yrs | 0.81 | -0.04 (-0.08-0.01) | 1.71 (p = 1.6e-6) | 1.89 (p = 1.1e-8) |
| **Lead-I Student** | 9.80 yrs | 14.54 (14.02-15.03) yrs | 0.68 | 0.01 (-0.04-0.05) | 1.53 (p = 8.7e-5) | 3.52 (p = 6.2e-29) |

## Reproducing Results
1. python src/preprocess.py
2. python src/train.py
3. python src/evaluate.py
4. python src/plot_figures.py

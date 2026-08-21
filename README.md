# PTB-XL Teacher–Student Knowledge Distillation

Reproducible research code for binary detection of **myocardial infarction (MI) or explicit ischemia** versus other ECG findings on PTB-XL, followed by a two-lead student trained with multi-level knowledge distillation.

## Data policy

This repository intentionally contains **no PTB-XL signals, metadata tables, checkpoints, virtual environments, caches, or other large generated files**. Download PTB-XL separately from PhysioNet and place it under `C:\ptbxl` (or adapt the configured root paths).

## Split and evaluation policy

- Training: official folds 1–8
- Validation/model selection: fold 9
- Final test: fold 10
- Patient-disjoint integrity is checked by `02_create_splits.py`.
- The test fold remains sealed until the teacher, threshold, and ensemble are finalized.

## Pipeline

1. `01_audit_dataset.py` — audit dataset files and labels.
2. `02_create_splits.py` — create patient-disjoint splits.
3. `03_train_teacher.py` — train the 500 Hz teacher.
4. `07_optimize_teacher_validation.py` — validation-only TTA/threshold search.
5. `08_finetune_teacher.py` — resumable weak-augmentation fine-tuning.
6. `09_fast_teacher_ensemble.py` — validation-only ensemble search.
7. `10_train_fast_teacher_100hz.py` — lightweight 100 Hz teacher experiment.
8. `11_train_official_xresnet.py` — adapted official PTB-XL XResNet1D101 experiment.
9. `04_train_student.py` — baseline/KD student training after teacher selection.
10. `05_evaluate_and_export.py` — one-time final test evaluation and exports.
11. `06_build_research_bundle.py` — aggregate research outputs.

## Windows setup

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_environment.ps1
.\run_pipeline.ps1
```

Long-running training scripts write resumable checkpoints approximately every five minutes. Generated artifacts belong in `results/` and model checkpoints are excluded from Git.

## Current validation status

The best completed configuration at the time of publication is the original/fine-tuned ensemble with validation Accuracy **87.769%** and AUC **93.225%**. The official XResNet1D101 adaptation is the current experiment. Fold 10 has not been evaluated.

See `results/ACTIVITY_REPORT.md` for the execution history, operational issues, decisions, and detailed tables.

## Self-hosted runner safety

The included workflow is manual-only and targets the label `ptbxl-local`. Do not add `pull_request` triggers to a public repository: untrusted workflow code must never execute on a personal laptop. The runner directory and credentials are ignored by Git.

## Third-party source

The adapted XResNet implementation under `external/official_ptbxl_models/` originates from the public PTB-XL benchmarking repository by Strodthoff et al. See `THIRD_PARTY_NOTICES.md` and the upstream license before redistribution or modification.


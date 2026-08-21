# ChatGPT Project setup: PTB-XL Teacher–Student KD

## Project name

`PTB-XL Teacher-Student KD`

## Repository

https://github.com/alimirzae/ptbxl-teacher-kd

## Project instructions

Paste the following text into **Project settings → Project instructions**:

> You are collaborating on a doctoral research project using PTB-XL for binary MI-or-ischemia detection and two-lead multi-level knowledge distillation. Treat GitHub repository `alimirzae/ptbxl-teacher-kd` as the source of truth for code and lightweight reports. Never commit or request raw PTB-XL records, metadata datasets, virtual environments, caches, model checkpoints, runner credentials, or other large artifacts. Preserve the official patient-disjoint split: folds 1–8 train, fold 9 validation, fold 10 final test. Do not use fold 10 for model selection, threshold tuning, or ensemble tuning. First finalize the Teacher on validation; then train the two-lead Student; evaluate Test only once after locking the model and threshold. Keep every experiment reproducible, resumable, and documented in JSON/CSV/Markdown. Report GPU activity honestly and distinguish active CUDA computation from CPU/disk data-loading waits. Before proposing a new run, read `README.md`, `results/ACTIVITY_REPORT.md`, and the latest experiment report. Use the manual GitHub Actions workflow only on the trusted self-hosted runner and never add public pull-request execution.

## Recommended project sources

Add these lightweight files from the repository as project context if direct repository access is unavailable:

1. `README.md`
2. `results/ACTIVITY_REPORT.md`
3. `results/ACTIVITY_REPORT.docx`
4. `results/TEACHER_DEVELOPMENT_LOG.json`
5. The three files under `doc/`

Do not upload datasets, raw ECG files, `.pt` checkpoints, `.npy` probability arrays, `.venv`, or `actions-runner`.

## First message for the new ChatGPT project

> Continue the PTB-XL Teacher–Student KD work from the public repository https://github.com/alimirzae/ptbxl-teacher-kd. Read README.md and results/ACTIVITY_REPORT.md first. Check the latest GitHub Actions status and the current `teacher_official_xresnet101` validation report. Keep fold 10 sealed. Continue toward a stronger Teacher, document every run, and only then proceed to the two-lead Student and knowledge distillation.


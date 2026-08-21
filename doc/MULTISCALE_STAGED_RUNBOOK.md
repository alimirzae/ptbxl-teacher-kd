# MultiScale Teacher — Staged Runner Runbook

## Current decision

The locked validation benchmark before this experiment is the original/fine-tuned ensemble: Accuracy 87.7691%, AUC 93.2248%. FastTeacher100Hz and the from-scratch official XResNet1D101 did not beat it. Fold 10 remains sealed.

The next Teacher experiment is a two-branch 500 Hz multi-scale model. One branch processes the native 500 Hz ECG and the other receives an internal 5x average-pooled view. Age and sex are fused only after ECG feature extraction.

## Why staged execution

The local self-hosted runner uses a Quadro P1000 and GitHub jobs must not depend on a long uninterrupted training process. Each workflow dispatch therefore has a 10-minute job timeout and the Python stage uses a default 420-second compute budget, leaving time for checkout, validation of prerequisites, state persistence and lightweight result publication.

Training and validation are both resumable. State is stored locally under `C:\ptbxl\results\teacher_multiscale_staged\training_latest.pt` and includes model/optimizer/scheduler/scaler state, phase, epoch, next training sample, next validation sample, accumulated loss, partial validation predictions and history. A later dispatch resumes from that state rather than replaying completed samples.

## Execution sequence

1. Run workflow `Staged Multiscale Teacher` with `mode=smoke`, `max_seconds=420`.
2. Smoke uses 80 train and 80 validation records and exactly one epoch. It must complete with `test_evaluated=false`.
3. Only after smoke succeeds, run the same workflow with `mode=train`, `max_seconds=420`.
4. If `stage_status.json` reports `run_status=needs_resume`, dispatch `mode=train` again. Do not reset local state.
5. After each completed validation epoch, `history.csv` and `report.json` are refreshed locally. The workflow also copies only lightweight `stage_status.json`, `history.csv`, and `report.json` into the repository and attempts to commit them to `main`.
6. Continue staged dispatches until one stop condition occurs: validation Accuracy >=90%, early stopping after at least 10 epochs and 7 stale epochs, or 24 epochs.
7. Do not evaluate fold 10 during this process.

## Files that must stay local

Do not commit `training_latest.pt`, `teacher_multiscale_best.pt`, `best_validation_probabilities.csv`, raw PTB-XL records, split source datasets, virtual environments, runner credentials or caches.

## Interpretation gate

The multi-scale model is considered a useful Teacher candidate only if it improves meaningfully over the locked 87.7691% validation Accuracy benchmark without using fold 10. If it fails to improve, the next preferred alternatives are a genuinely ECG-pretrained backbone or ending Teacher architecture search and proceeding to the two-lead Student/KD phase with the locked ensemble Teacher.

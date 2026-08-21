# MultiScaleTeacher500Hz experiment plan

## Motivation

The current best validation result remains the original/fine-tuned ensemble at 87.7691% accuracy and 93.2248% AUC. FastTeacher100Hz reached 86.2574% accuracy, and the adapted official XResNet1D101 trained from scratch reached 85.1580%. The next experiment therefore changes the representation rather than repeating threshold tuning or another similar single-scale backbone.

## Hypothesis

A two-branch 500 Hz teacher can preserve fine ECG morphology in a high-resolution branch while simultaneously modelling broader temporal structure in a branch downsampled internally to 100 Hz. Age and sex are fused only after ECG feature extraction. This may improve validation discrimination without using any fold-10 information.

## Data policy

- Train: official folds 1–8 only.
- Validation/model selection/threshold tuning: fold 9 only.
- Test: fold 10 remains sealed and is not loaded by `12_train_multiscale_teacher.py`.
- No raw ECG, metadata table, checkpoint, probability array, virtual environment, cache, or runner credential is committed to Git.

## Model

`12_train_multiscale_teacher.py` implements `MultiScaleTeacher500Hz`:

- high-resolution branch: 500 Hz input;
- low-resolution branch: average-pooled 100 Hz view derived inside the model from the same 500 Hz ECG;
- residual convolutional blocks with squeeze-and-excitation and increasing dilation;
- mean/max temporal pooling for each branch;
- late fusion of age and sex;
- binary output with validation-only threshold search.

The training path uses AdamW, OneCycleLR, mixed precision when CUDA is available, gradient clipping, weak augmentation, five-minute atomic checkpoints, deterministic epoch-level shuffling, resumable state, CSV history, JSON report and saved validation probabilities for the best epoch.

## Execution order

1. Run a smoke test first:

   `C:\Python\Python312\python.exe 12_train_multiscale_teacher.py --mode smoke`

2. Inspect that the smoke run completes forward/backward/validation, writes `C:\ptbxl\results\teacher_multiscale_500hz\report.json`, and explicitly reports `test_evaluated=false`.

3. Only after the smoke test passes, run the full validation-only experiment:

   `C:\Python\Python312\python.exe 12_train_multiscale_teacher.py --mode train`

4. Compare the best fold-9 accuracy/AUC against the locked current benchmark of 87.7691% / 93.2248%.

5. Do not evaluate fold 10 even if the multiscale model exceeds 90%. First document the run, lock the selected Teacher/checkpoint and threshold, then proceed to the Student/KD stage according to the project protocol.

## Decision criteria

- Primary: validation accuracy.
- Secondary: AUC, sensitivity, specificity and F1.
- Stop early after a sustained validation plateau according to the script patience setting.
- If the model fails to beat 87.7691%, stop this branch rather than repeatedly tuning fold 9.
- If it clearly beats the current Teacher, document the result and consider a validation-only ensemble with the existing best Teacher before locking the final Teacher.

## Alternatives after this run

If the multiscale model does not improve validation, the preferred next alternatives are: (1) a genuinely pretrained ECG representation with externally sourced weights and provenance documented before fine-tuning, or (2) moving forward with the current best Teacher and focusing research effort on multi-level knowledge distillation rather than further repeated validation optimization.

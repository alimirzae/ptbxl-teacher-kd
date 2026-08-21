
# DiabHeartEdge — Chapter 3 Implementation Specification
## Autonomous Pipeline: Teacher Training → Knowledge Distillation → Student Training
### Binary ECG Classification (Myocardial Infarction + Ischemia vs. Others) on PTB-XL

---

> ## ⚠️ AGENT DIRECTIVE (READ FIRST)
> You are an autonomous senior ML engineer. Execute this ENTIRE specification
> end-to-end WITHOUT asking questions. If any step fails, debug and fix it
> yourself. NEVER fabricate, hardcode, or artificially inflate results —
> all numbers must come from real evaluation on the held-out test set.
> Produce every artifact listed in §10. Log everything.

---

## 0. MISSION SUMMARY

| Item | Specification |
|------|---------------|
| Dataset | PTB-XL v1.0.3 (PhysioNet), `records500` (500 Hz, 10 s, 12 leads) |
| Task | Binary classification: **POSITIVE = {class 0 (MI), class 1 (ISCHEMIA)}** vs NEGATIVE = {classes 2–8} |
| Clinical meaning | Positive = heart-attack spectrum (infarction + ischemia) |
| Teacher | 12-lead SE-ResNet + Transformer (~26M params) |
| Student | 2-lead (Lead II + aVF) lightweight CNN-Transformer (~0.5M params) |
| Distillation | Multi-level: Logit KD + Feature KD + Hard CE |
| Targets | Teacher test Acc ≥ 90% · Student gap ≤ 3% · Compression ≥ 20x · Positive Sensitivity ≥ 85% |

---

## 1. DATASET & SPLITS (DO NOT MODIFY)

- Total: 21,799 records / 18,869 patients.
- Provided CSVs: `train.csv`, `val.csv`, `test.csv` with columns:
  `ecg_id, patient_id, age, sex, ..., strat_fold, filename_lr, filename_hr, final_label, label, class_id`
- **Splits are patient-disjoint and FIXED:** folds 1–8 = train, fold 9 = val, fold 10 = test.
  NEVER re-split, shuffle across splits, or use test data for training/selection.
- 9-class mapping (column `class_id`):
  `0=MI, 1=ISCHEMIA, 2=LBBB, 3=OTHER_CONDUCTION, 4=PVC, 5=AFIB, 6=OTHER_ARRHYTHMIA, 7=HYPERTROPHY, 8=NORMAL`
- **Binary label:** `y = 1 if class_id in {0,1} else 0`  (≈31.4% positive).
- Signals: load via `wfdb.rdsamp(<ptbxl_root>/<filename_hr>)` → shape (5000, 12).
- Lead order: `[I, II, III, aVR, aVL, aVF, V1..V6]` → **Student uses indices [1, 5] (II, aVF).**
- Missing/corrupt files: skip gracefully and LOG the count (expect ~2%).

## 2. PREPROCESSING & AUGMENTATION

- Per-lead Z-score normalization: `(x - mean)/std` per channel.
- **Train-only augmentation** (apply stochastically): Gaussian noise (SNR 12–30 dB),
  baseline wander (0.1–0.6 Hz sine, amp 0.05–0.25), amplitude scaling (0.8–1.2),
  random lead dropout (1–2 leads, p=0.3), time shift (±500 samples, p=0.5),
  cutout (250–1000 samples zeroed, p=0.3).
- Validation/Test: NO augmentation (except TTA at evaluation, §7).

## 3. TEACHER ARCHITECTURE (12-lead)

Stem: Conv1d(12→64, k15, s2)+BN+ReLU →
Stage1: ResBlock(64→128, s2)+ResBlock(128) →
Stage2: ResBlock(128→256, s2)+ResBlock(256) →
Stage3: ResBlock(256→512, s2)+ResBlock(512) →
(ResBlock = 2×[Conv k7 + BN] + Squeeze-Excitation + residual) →
AdaptiveAvgPool1d(50) → TransformerEncoder(d=512, heads=8, ff=1024, layers=2, drop=0.2) →
Classifier: Linear(512·50→512)+GELU+Drop(0.4)+Linear(512→2).

## 4. STUDENT ARCHITECTURE (2-lead: II + aVF)

Conv1d(2→32, k7, s2)+BN+ReLU → Conv1d(32→64, k9, s2)+BN+ReLU →
Conv1d(64→128, k11, s2)+BN+ReLU → AdaptiveAvgPool1d(50) →
TransformerEncoder(d=128, heads=4, ff=256, layers=2, drop=0.1) → mean-pool →
Classifier: Linear(128→128)+GELU+Drop(0.3)+Linear(128→2).
**Feature projection for KD:** Conv1d(128→512, k1) to match Teacher feature map (B,512,50).

## 5. TEACHER TRAINING PROTOCOL

- Seed 42 everywhere; `cudnn.benchmark=True`.
- Loss: CrossEntropy with **balanced class weights** + label_smoothing=0.05;
  **Mixup** p=0.5, λ~Beta(0.2,0.2), loss = λ·CE(a)+(1−λ)·CE(b).
- Optimizer: AdamW, lr=3e-4, wd=1e-4; 1-epoch linear warmup + cosine decay (min 5% lr).
- Epochs=15, Early stopping patience=5 on **val accuracy**.
- **EMA** decay=0.999 — update ONLY floating-point tensors; copy integer buffers
  (e.g. BatchNorm `num_batches_tracked`) verbatim. Use EMA weights for evaluation/saving.
- **AMP** (GradScaler + autocast) with version-safe fallback API.
- Batch 32 (train) / 64 (eval); num_workers=2; gradient clip norm 1.0.
- Save best EMA checkpoint: `checkpoints/teacher/teacher_mi_binary_best.pt`
  (include val_acc, test metrics, chosen threshold).

## 6. KNOWLEDGE DISTILLATION → STUDENT TRAINING

- Freeze Teacher (eval mode, `requires_grad=False`).
- Student input = leads [1,5]; Teacher input = all 12 leads.
- Total loss = **CE_hard(student, y, weighted)**
  + α·T²·KL(log_softmax(s/T) ‖ softmax(t/T)), T=2.0, α=0.7
  + β·MSE(proj(student_feat), teacher_feat), β=0.3.
- Same optimizer/schedule/EMA/AMP/early-stopping as §5.
- ALSO train a **baseline Student (CE only, USE_KD=False)** for ablation.
- Save: `checkpoints/student/student_mi_kd_best.pt` and `student_baseline_best.pt`.

## 7. EVALUATION PROTOCOL (HONEST — NON-NEGOTIABLE)

1. Model selection & threshold tuning on **val ONLY** (sweep 0.20–0.80, step 0.01, maximize Acc).
2. **Test set evaluated EXACTLY ONCE** at the end with the fixed threshold.
3. **TTA** at evaluation: 1 clean + 4 augmented passes (mild noise/scale/shift), average softmax.
4. Metrics to report (val AND test): Accuracy, Positive Sensitivity, Specificity,
   F1 (positive), AUC, confusion matrix, ROC curve; include always-negative baseline.
5. Report Teacher-vs-Student comparison table + parameter counts + compression ratio.

## 8. SCIENTIFIC INTEGRITY RULES

- No leakage: never touch test during training/selection; keep patient-disjoint splits.
- No fabricated or placeholder metrics; no threshold tuning on test.
- Reproducible: fixed seeds, save configs, log all hyperparameters.
- If targets (§0) are not met, iterate legitimately (hyperparameter search over
  lr∈{1e-4,3e-4}, α∈{0.5,0.7,1.0}, β∈{0.2,0.3,0.5}, T∈{2,3,4}, epochs≤25) and report the best honest run.

## 9. ENVIRONMENT & PATHS (CONFIG)

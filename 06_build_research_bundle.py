import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\ptbxl")
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_records(path):
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def relative_files(folder):
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted(folder.rglob("*")) if p.is_file()]


def main():
    metrics_path = RAW / "metrics.json"
    comparison_path = TABLES / "model_comparison.csv"
    distribution_path = TABLES / "dataset_distribution.csv"
    required = [metrics_path, comparison_path, distribution_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Evaluation outputs are missing: " + ", ".join(missing))

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = csv_records(comparison_path)
    distribution = csv_records(distribution_path)
    histories = {}
    history_paths = {
        "teacher": ROOT / "project/runs/teacher/train_history.csv",
        "student_baseline": ROOT / "project/runs/student/baseline/history.csv",
        "student_kd": ROOT / "project/runs/student/kd/history.csv",
    }
    for name, path in history_paths.items():
        histories[name] = csv_records(path) if path.exists() else []

    source_files = [
        ROOT / "01_audit_dataset.py", ROOT / "02_create_splits.py",
        ROOT / "03_train_teacher.py", ROOT / "04_train_student.py",
        ROOT / "05_evaluate_and_export.py", ROOT / "06_build_research_bundle.py",
        ROOT / "ptbxl_database.csv", ROOT / "scp_statements.csv",
    ]
    bundle = {
        "schema": {
            "name": "DiabHeartEdge thesis experiment bundle",
            "version": "1.0.0",
            "language": "fa",
            "purpose": "A self-contained evidence package for drafting thesis Chapters 3, 4, and 5 without inventing results.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "scientific_scope": {
            "dataset": "PTB-XL v1.0.3",
            "task": "Binary ECG classification: myocardial infarction or explicit ischemia versus all other records",
            "positive_definition": "Any SCP code in PTB-XL diagnostic class MI, plus explicit ischemia codes ISC*.",
            "negative_definition": "All records without an MI or explicit ischemia code.",
            "teacher": "12-lead SE-ResNet + 2-layer Transformer encoder",
            "student": "2-lead (II and aVF) lightweight CNN + 2-layer Transformer encoder",
            "distillation": "Hard weighted cross-entropy + logit KL distillation + feature-map MSE",
            "primary_endpoint": "Held-out test accuracy",
            "secondary_endpoints": ["sensitivity", "specificity", "positive-class F1", "ROC AUC"],
        },
        "integrity_protocol": {
            "split_rule": "PTB-XL folds 1-8 train, fold 9 validation, fold 10 test",
            "patient_disjoint": True,
            "threshold_selection": "Validation set only; grid 0.20 to 0.80 in steps of 0.01 maximizing accuracy",
            "test_access": "Exactly one final evaluation after all model and threshold selection",
            "test_time_augmentation": "One clean pass plus four mild scale/noise/time-shift passes; mean positive softmax probability",
            "uncertainty": "95% percentile bootstrap confidence intervals with 1000 resamples and seed 42",
            "warning": "Do not treat smoke-test metrics as scientific results. Only values under final_metrics are reportable.",
        },
        "dataset_distribution": distribution,
        "final_metrics": metrics,
        "comparison_table": comparison,
        "training_histories": histories,
        "figure_catalog": {
            "roc_comparison.png": "ROC curves on the held-out test set for all three models.",
            "precision_recall_comparison.png": "Precision-recall curves on the held-out test set.",
            "metrics_comparison.png": "Grouped comparison of accuracy, sensitivity, specificity, F1, and AUC.",
            "confusion_matrices.png": "Test confusion matrices using validation-selected thresholds.",
            "calibration_comparison.png": "Reliability curves comparing predicted probability and observed prevalence.",
        },
        "chapter_drafting_guide": {
            "chapter_3": [
                "Dataset and inclusion criteria", "Label derivation", "Patient-disjoint split",
                "Preprocessing and augmentation", "Teacher architecture", "Student architecture",
                "Knowledge-distillation objective", "Optimization", "Honest evaluation protocol",
                "Hardware/software and reproducibility"
            ],
            "chapter_4": [
                "Dataset statistics", "Training behavior", "Validation threshold selection",
                "Held-out test metrics with confidence intervals", "ROC and PR analysis",
                "Confusion matrices", "Compression and ablation", "Calibration"
            ],
            "chapter_5": [
                "Interpretation of findings", "Clinical and edge-device implications",
                "Teacher-student trade-offs", "Comparison with stated targets",
                "Limitations", "Threats to validity", "Future research", "Conclusion"
            ],
            "mandatory_rule": "Use only final_metrics and comparison_table for numerical claims; explicitly label unmet targets and limitations.",
        },
        "environment": {
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__, "pandas": pd.__version__, "seed": 42,
        },
        "artifacts": relative_files(RESULTS),
        "provenance": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                       for p in source_files if p.exists()},
    }
    json_path = RESULTS / "thesis_results_bundle.json"
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    comp_df = pd.DataFrame(comparison)
    dist_df = pd.DataFrame(distribution)
    lines = [
        "# بستهٔ نتایج پژوهش DiabHeartEdge",
        "",
        "> این فایل به‌صورت خودکار از خروجی‌های واقعی pipeline ساخته شده است. اعداد Smoke Test در آن گزارش علمی محسوب نمی‌شوند.",
        "",
        "## تعریف مسئله",
        "",
        "طبقه‌بندی دودویی ECG دیتاست PTB-XL: وجود انفارکتوس میوکارد یا ایسکمی صریح در برابر سایر رکوردها. Teacher از ۱۲ لید و Student از لیدهای II و aVF استفاده می‌کند.",
        "",
        "## پروتکل جلوگیری از نشت داده",
        "",
        "- folds 1 تا 8 برای آموزش، fold 9 برای اعتبارسنجی و fold 10 برای آزمون نهایی است.",
        "- بیماران بین مجموعه‌ها مشترک نیستند.",
        "- انتخاب مدل و آستانه فقط روی validation انجام شده است.",
        "- test فقط پس از تثبیت مدل‌ها و آستانه‌ها ارزیابی شده است.",
        "- بازه‌های اطمینان ۹۵٪ با ۱۰۰۰ bootstrap و seed=42 محاسبه شده‌اند.",
        "",
        "## توزیع داده‌ها", "", dist_df.to_markdown(index=False), "",
        "## جدول اصلی نتایج آزمون", "", comp_df.to_markdown(index=False), "",
        "## فایل‌های نمودار", "",
    ]
    for name, caption in bundle["figure_catalog"].items():
        lines += [f"### {caption}", "", f"![{caption}](figures/{name})", ""]
    lines += [
        "## راهنمای نگارش فصل سوم", "",
        "فصل سوم باید داده، تعریف برچسب، پیش‌پردازش، معماری‌ها، تابع زیان KD، جزئیات آموزش و پروتکل ارزیابی را بازتولیدپذیر بیان کند.", "",
        "## راهنمای نگارش فصل چهارم", "",
        "فصل چهارم باید جدول اصلی نتایج، بازه‌های اطمینان، منحنی‌های ROC و PR، ماتریس‌های درهم‌ریختگی، calibration و مقایسهٔ فشرده‌سازی را بدون افزودن عدد ساختگی گزارش کند.", "",
        "## راهنمای نگارش فصل پنجم", "",
        "فصل پنجم باید نتایج را تفسیر کند، اهداف از پیش تعیین‌شده را با اعداد واقعی مقایسه کند و محدودیت‌های PTB-XL، برچسب‌گذاری، تعمیم‌پذیری، دو لید و سخت‌افزار را صریحاً توضیح دهد.", "",
        "## راهنمای استفاده با مدل‌های هوش مصنوعی", "",
        "فایل `thesis_results_bundle.json` را به مدل بدهید و تأکید کنید تنها از `final_metrics` و `comparison_table` برای ادعاهای عددی استفاده کند. برای متن خوانا و مسیر تصاویر، همین فایل Markdown نیز همراه آن ارائه شود.", "",
        "## بازتولید", "",
        "در PowerShell ویندوز ابتدا `setup_environment.ps1` و سپس `run_pipeline.ps1` اجرا شود. تمام logها در `results/logs` نگهداری می‌شوند.", "",
    ]
    md_path = RESULTS / "thesis_results_context.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {json_path}")
    print(f"Created {md_path}")


if __name__ == "__main__":
    main()

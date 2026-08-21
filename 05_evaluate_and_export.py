import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ROOT = Path(r"C:\ptbxl")
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
RAW = RESULTS / "raw"
for p in (FIGURES, TABLES, RAW): p.mkdir(parents=True, exist_ok=True)


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


teacher_mod = load_module("teacher_pipeline", ROOT / "03_train_teacher.py")
student_mod = load_module("student_pipeline", ROOT / "04_train_student.py")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast = teacher_mod.make_autocast(device)


def predict_teacher(model, loader, threshold):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast():
                passes = [torch.softmax(model(x), 1)[:, 1]]
                for _ in range(4):
                    passes.append(torch.softmax(model(teacher_mod.tta_aug(x)), 1)[:, 1])
            ps.append(torch.stack(passes).mean(0).cpu().numpy()); ys.append(y.numpy())
    return summarize(np.concatenate(ys), np.concatenate(ps), threshold)


def predict_student(model, loader, threshold):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)[:, student_mod.STUDENT_LEADS]
            with autocast():
                passes = [torch.softmax(model(x)[0], 1)[:, 1]]
                for _ in range(4):
                    passes.append(torch.softmax(model(teacher_mod.tta_aug(x))[0], 1)[:, 1])
            ps.append(torch.stack(passes).mean(0).cpu().numpy()); ys.append(y.numpy())
    return summarize(np.concatenate(ys), np.concatenate(ps), threshold)


def summarize(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    metrics = {
        "threshold": float(threshold), "n": int(len(y)),
        "accuracy": float(100 * (pred == y).mean()),
        "sensitivity": float(100 * tp / max(1, tp + fn)),
        "specificity": float(100 * tn / max(1, tn + fp)),
        "f1": float(100 * 2 * tp / max(1, 2 * tp + fp + fn)),
        "auc": float(100 * roc_auc_score(y, p)),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "always_negative_accuracy": float(100 * (y == 0).mean()),
    }
    return {"metrics": metrics, "targets": y, "probabilities": p, "predictions": pred}


def bootstrap_ci(y, p, threshold, repeats=1000):
    rng = np.random.default_rng(42); rows = []
    n = len(y)
    for _ in range(repeats):
        idx = rng.integers(0, n, n); yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2: continue
        rows.append(summarize(yy, pp, threshold)["metrics"])
    out = {}
    for key in ["accuracy", "sensitivity", "specificity", "f1", "auc"]:
        values = np.array([r[key] for r in rows])
        out[key] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return out


def main():
    split = ROOT / "project" / "data" / "splits"
    val_ds = teacher_mod.ECGDataset(split / "val.csv", ROOT, augment=False)
    test_ds = teacher_mod.ECGDataset(split / "test.csv", ROOT, augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    specs = [
        ("Teacher (12-lead)", "teacher", ROOT/"project/checkpoints/teacher/teacher_mi_binary_best.pt"),
        ("Student baseline (2-lead)", "student", ROOT/"project/checkpoints/student/student_baseline_best.pt"),
        ("Student KD (2-lead)", "student", ROOT/"project/checkpoints/student/student_mi_kd_best.pt"),
    ]
    results = {}
    for name, kind, path in specs:
        ckpt = torch.load(path, map_location=device); threshold = float(ckpt["threshold"])
        if kind == "teacher":
            model = teacher_mod.TeacherBinary().to(device); model.load_state_dict(ckpt["model_state_dict"])
            val = predict_teacher(model, val_loader, threshold)
            test = predict_teacher(model, test_loader, threshold)
        else:
            model = student_mod.StudentLite().to(device); model.load_state_dict(ckpt["model_state_dict"])
            val = predict_student(model, val_loader, threshold)
            test = predict_student(model, test_loader, threshold)
        test["metrics"]["ci95"] = bootstrap_ci(test["targets"], test["probabilities"], threshold)
        results[name] = {"val": val, "test": test,
                         "parameters": sum(p.numel() for p in model.parameters())}
        pd.DataFrame({"target": test["targets"], "probability": test["probabilities"],
                      "prediction": test["predictions"]}).to_csv(
                          RAW / f"{kind}_{'kd' if 'KD' in name else 'baseline' if 'baseline' in name else '12lead'}_test_predictions.csv",
                          index=False)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    rows = []
    for name, r in results.items():
        m = r["test"]["metrics"]; ci = m["ci95"]
        rows.append({"Model": name, "Parameters": r["parameters"],
                     "Threshold": m["threshold"], "Accuracy (%)": m["accuracy"],
                     "Accuracy 95% CI": f"{ci['accuracy'][0]:.2f}-{ci['accuracy'][1]:.2f}",
                     "Sensitivity (%)": m["sensitivity"], "Specificity (%)": m["specificity"],
                     "F1 (%)": m["f1"], "AUC (%)": m["auc"],
                     "TP": m["tp"], "FN": m["fn"], "TN": m["tn"], "FP": m["fp"]})
    summary = pd.DataFrame(rows)
    teacher_params = summary.iloc[0]["Parameters"]
    summary["Compression (x)"] = teacher_params / summary["Parameters"]
    summary.to_csv(TABLES / "model_comparison.csv", index=False)

    split_rows = []
    for name in ["train", "val", "test"]:
        d = pd.read_csv(split / f"{name}.csv")
        split_rows.append({"Split": name, "Total": len(d), "Positive": int(d.class_id.sum()),
                           "Negative": int((d.class_id == 0).sum()),
                           "Positive (%)": float(100*d.class_id.mean()),
                           "Patients": int(d.patient_id.nunique())})
    pd.DataFrame(split_rows).to_csv(TABLES / "dataset_distribution.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, r), c in zip(results.items(), colors):
        y, p = r["test"]["targets"], r["test"]["probabilities"]
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, lw=2, color=c,
                                               label=f"{name} (AUC={roc_auc_score(y,p):.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=.5); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Test ROC Curves")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES/"roc_comparison.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, r), c in zip(results.items(), colors):
        y, p = r["test"]["targets"], r["test"]["probabilities"]
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, lw=2, color=c, label=f"{name} (AUC={auc(recall,precision):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test Precision-Recall Curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES/"precision_recall_comparison.png", dpi=300); plt.close(fig)

    metric_cols = ["Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "F1 (%)", "AUC (%)"]
    x = np.arange(len(metric_cols)); width = .25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, row in summary.iterrows(): ax.bar(x+(i-1)*width, row[metric_cols], width, label=row["Model"], color=colors[i])
    ax.set_xticks(x, [c.replace(" (%)", "") for c in metric_cols]); ax.set_ylim(0,100); ax.set_ylabel("Percent")
    ax.set_title("Held-out Test Performance"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIGURES/"metrics_comparison.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, ((name, r), c) in zip(axes, zip(results.items(), colors)):
        m = r["test"]["metrics"]; cm = np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]])
        im=ax.imshow(cm,cmap="Blues");
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center",fontsize=12)
        ax.set_xticks([0,1],["Negative","Positive"]); ax.set_yticks([0,1],["Negative","Positive"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES/"confusion_matrices.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,6))
    for (name,r),c in zip(results.items(),colors):
        frac, mean = calibration_curve(r["test"]["targets"], r["test"]["probabilities"], n_bins=10)
        ax.plot(mean, frac, marker="o", color=c, label=name)
    ax.plot([0,1],[0,1],"k--"); ax.set(xlabel="Mean predicted probability",ylabel="Observed positive fraction",title="Calibration Curves")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES/"calibration_comparison.png",dpi=300); plt.close(fig)

    serializable = {}
    for name, r in results.items():
        serializable[name] = {"parameters": r["parameters"],
                              "validation": r["val"]["metrics"], "test": r["test"]["metrics"]}
    (RAW/"metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(f"Exported to {RESULTS}")


if __name__ == "__main__":
    main()

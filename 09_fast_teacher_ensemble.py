import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(r"C:\ptbxl")
OUT = ROOT / "results" / "teacher_optimization" / "fast_ensemble"
OUT.mkdir(parents=True, exist_ok=True)


def load_teacher_module():
    spec = spec_from_file_location("teacher_pipeline", ROOT / "03_train_teacher.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    from sklearn.metrics import roc_auc_score
    return {
        "threshold": float(threshold), "accuracy": float(100 * (pred == y).mean()),
        "sensitivity": float(100 * tp / max(1, tp + fn)),
        "specificity": float(100 * tn / max(1, tn + fp)),
        "f1": float(100 * 2 * tp / max(1, 2 * tp + fp + fn)),
        "auc": float(100 * roc_auc_score(y, p)), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def tune(y, p):
    return max((metrics(y, p, t) for t in np.arange(.05, .951, .001)),
               key=lambda r: (r["accuracy"], r["sensitivity"], r["f1"]))


@torch.no_grad()
def predict(model, loader, device, autocast, passes=4):
    model.eval(); ys = []; variants = [[] for _ in range(passes + 1)]
    torch.manual_seed(42); np.random.seed(42)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            variants[0].append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
            for i in range(1, passes + 1):
                variants[i].append(torch.softmax(model(teacher.tta_aug(x)), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    arrays = [np.concatenate(v) for v in variants]
    return np.concatenate(ys), arrays


teacher = load_teacher_module()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast = teacher.make_autocast(device)
ds = teacher.ECGDataset(ROOT / "project/data/splits/val.csv", ROOT, augment=False)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

model = teacher.TeacherBinary().to(device)
checkpoint = torch.load(
    ROOT / "results/teacher_optimization/finetune_weak_aug/teacher_finetuned_best.pt",
    map_location=device,
)
model.load_state_dict(checkpoint["model_state_dict"])
y, fine_variants = predict(model, loader, device, autocast, passes=4)

original = {}
for tta in (0, 1, 2, 4):
    frame = pd.read_csv(ROOT / f"results/teacher_optimization/validation_probabilities_tta_{tta}.csv")
    assert np.array_equal(y, frame["target"].to_numpy())
    original[f"original_tta_{tta}"] = frame["probability"].to_numpy()

candidates = dict(original)
for n in range(5):
    candidates[f"finetuned_tta_{n}"] = np.mean(fine_variants[:n + 1], axis=0)

rows = []
for name, p in candidates.items():
    rows.append({"name": name, "blend_weight_finetuned": None, **tune(y, p)})
for oname, op in original.items():
    for n in range(5):
        fp = candidates[f"finetuned_tta_{n}"]
        for weight in np.arange(.05, 1.0, .05):
            p = weight * fp + (1 - weight) * op
            rows.append({"name": f"{oname}+finetuned_tta_{n}",
                         "blend_weight_finetuned": round(float(weight), 2), **tune(y, p)})

table = pd.DataFrame(rows).sort_values(["accuracy", "auc", "sensitivity"], ascending=False)
table.to_csv(OUT / "ensemble_validation_results.csv", index=False)
best = table.iloc[0].to_dict()
pd.DataFrame({"target": y, **candidates}).to_csv(OUT / "validation_probabilities.csv", index=False)
report = {"scope": "validation_only", "test_evaluated": False, "target_accuracy": 90.0,
          "target_reached": bool(best["accuracy"] >= 90.0), "best": best,
          "evaluated_candidates": int(len(table))}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("TEST SET NOT LOADED OR EVALUATED")

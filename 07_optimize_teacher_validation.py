import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(r"C:\ptbxl")
OUT = ROOT / "results" / "teacher_optimization"
OUT.mkdir(parents=True, exist_ok=True)


def load_teacher_module():
    spec = spec_from_file_location("teacher_pipeline", ROOT / "03_train_teacher.py")
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": float(threshold), "accuracy": float(100 * (pred == y).mean()),
        "sensitivity": float(100 * tp / max(1, tp + fn)),
        "specificity": float(100 * tn / max(1, tn + fp)),
        "f1": float(100 * 2 * tp / max(1, 2 * tp + fp + fn)),
        "auc": float(100 * roc_auc_score(y, p)),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def sweep(y, p, require_sensitivity=None):
    best = None
    for threshold in np.arange(0.05, 0.951, 0.001):
        row = metrics(y, p, round(float(threshold), 3))
        if require_sensitivity is not None and row["sensitivity"] < require_sensitivity:
            continue
        score = (row["accuracy"], row["sensitivity"], row["f1"])
        if best is None or score > best[0]: best = (score, row)
    return best[1] if best else None


@torch.no_grad()
def predict(model, loader, device, autocast, teacher_mod, tta_passes):
    model.eval(); probs, targets = [], []
    torch.manual_seed(42); np.random.seed(42)
    for x, y in tqdm(loader, desc=f"Validation TTA={tta_passes}"):
        x = x.to(device, non_blocking=True)
        with autocast():
            passes = [torch.softmax(model(x), 1)[:, 1]]
            for _ in range(tta_passes):
                passes.append(torch.softmax(model(teacher_mod.tta_aug(x)), 1)[:, 1])
        probs.append(torch.stack(passes).mean(0).cpu().numpy()); targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(probs)


def main():
    teacher_mod = load_teacher_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = teacher_mod.make_autocast(device)
    dataset = teacher_mod.ECGDataset(
        ROOT / "project/data/splits/val.csv", ROOT, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    checkpoint_path = ROOT / "project/checkpoints/teacher/teacher_mi_binary_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = teacher_mod.TeacherBinary().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows, detail = [], {}
    for tta in [0, 1, 2, 4]:
        y, p = predict(model, loader, device, autocast, teacher_mod, tta)
        best_accuracy = sweep(y, p)
        best_clinical = sweep(y, p, require_sensitivity=85.0)
        detail[f"tta_{tta}"] = {
            "best_accuracy": best_accuracy,
            "best_with_sensitivity_at_least_85": best_clinical,
        }
        for selection, result in [("max_accuracy", best_accuracy),
                                  ("max_accuracy_sensitivity_ge_85", best_clinical)]:
            if result is not None:
                rows.append({"tta_passes": tta, "selection": selection, **result})
        pd.DataFrame({"target": y, "probability": p}).to_csv(
            OUT / f"validation_probabilities_tta_{tta}.csv", index=False)

    table = pd.DataFrame(rows).sort_values(["selection", "accuracy"], ascending=[True, False])
    table.to_csv(OUT / "inference_threshold_comparison.csv", index=False)
    best = table[table.selection == "max_accuracy"].iloc[0].to_dict()
    report = {
        "scope": "validation_only", "test_evaluated": False,
        "checkpoint": str(checkpoint_path), "threshold_step": 0.001,
        "tta_variants": [0, 1, 2, 4], "results": detail,
        "best_configuration": best,
        "target_accuracy_percent": 90.0,
        "target_reached": bool(best["accuracy"] > 90.0),
    }
    (OUT / "teacher_validation_optimization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(report["best_configuration"], indent=2))
    print("TEST SET NOT LOADED OR EVALUATED")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from importlib.util import module_from_spec, spec_from_file_location

SEED = 42
STUDENT_LEADS = [1, 5]


def load_teacher_module(root):
    spec = spec_from_file_location("teacher_pipeline", root / "03_train_teacher.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudentLite(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.c1 = nn.Conv1d(2, 32, 7, stride=2, padding=3)
        self.b1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(32, 64, 9, stride=2, padding=4)
        self.b2 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(64, 128, 11, stride=2, padding=5)
        self.b3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(50)
        enc = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.feat_proj = nn.Conv1d(128, 512, 1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        feat = self.pool(x)
        h = self.transformer(feat.permute(0, 2, 1)).mean(dim=1)
        return self.classifier(h), self.feat_proj(feat)


def teacher_forward_features(model, x):
    x = model.stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    feat = model.pool(x)
    h = model.transformer(feat.permute(0, 2, 1))
    logits = model.classifier(h.reshape(h.size(0), -1))
    return logits, feat


@torch.no_grad()
def evaluate(model, loader, device, autocast, teacher_mod, tta=0, threshold=0.5):
    model.eval()
    all_p, all_y = [], []
    for x, y in tqdm(loader, desc="Validation", leave=False):
        x = x.to(device, non_blocking=True)[:, STUDENT_LEADS]
        with autocast():
            probs = [torch.softmax(model(x)[0], 1)[:, 1]]
            for _ in range(tta):
                probs.append(torch.softmax(model(teacher_mod.tta_aug(x))[0], 1)[:, 1])
        all_p.append(torch.stack(probs).mean(0).cpu().numpy())
        all_y.append(y.numpy())
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    from sklearn.metrics import roc_auc_score
    return {
        "accuracy": float(100 * (pred == y).mean()),
        "sensitivity": float(100 * tp / max(1, tp + fn)),
        "specificity": float(100 * tn / max(1, tn + fp)),
        "f1": float(100 * 2 * tp / max(1, 2 * tp + fp + fn)),
        "auc": float(100 * roc_auc_score(y, p)),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "probabilities": p, "targets": y,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"C:\ptbxl"))
    parser.add_argument("--mode", choices=["baseline", "kd"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    teacher_mod = load_teacher_module(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = teacher_mod.make_autocast(device)
    scaler = teacher_mod.make_scaler(device)

    split_dir = args.root / "project" / "data" / "splits"
    run_dir = args.root / "project" / "runs" / "student" / args.mode
    ckpt_dir = args.root / "project" / "checkpoints" / "student"
    run_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_ds = teacher_mod.ECGDataset(split_dir / "train.csv", args.root, augment=True)
    val_ds = teacher_mod.ECGDataset(split_dir / "val.csv", args.root, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    n_pos = int(train_ds.df["class_id"].sum()); n_neg = len(train_ds) - n_pos
    weights = torch.tensor([len(train_ds)/(2*n_neg), len(train_ds)/(2*n_pos)],
                           dtype=torch.float32, device=device)
    student = StudentLite().to(device)
    n_params = sum(p.numel() for p in student.parameters())

    teacher = None
    if args.mode == "kd":
        teacher = teacher_mod.TeacherBinary().to(device)
        teacher_ckpt = torch.load(
            args.root / "project" / "checkpoints" / "teacher" / "teacher_mi_binary_best.pt",
            map_location=device,
        )
        teacher.load_state_dict(teacher_ckpt["model_state_dict"])
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(epoch):
        if epoch < 1: return epoch + 1
        progress = (epoch - 1) / max(1, args.epochs - 1)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = teacher_mod.EMA(student, decay=0.999)
    history, best_state, best_acc, stale = [], None, -1.0, 0
    start_epoch = 0
    snapshot_path = ckpt_dir / f"student_{args.mode}_training_latest.pt"
    if args.resume and snapshot_path.exists():
        snapshot = torch.load(snapshot_path, map_location=device)
        student.load_state_dict(snapshot["model_state_dict"])
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        scheduler.load_state_dict(snapshot["scheduler_state_dict"])
        ema.state = {k: v.to(device) for k, v in snapshot["ema_state_dict"].items()}
        best_state = snapshot.get("best_state")
        best_acc = float(snapshot.get("best_acc", -1.0))
        stale = int(snapshot.get("stale", 0))
        history = snapshot.get("history", [])
        start_epoch = int(snapshot["epoch"]) + 1
        print(f"Resuming {args.mode} from epoch {start_epoch} of {args.epochs}.")
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
        student.train(); total = 0.0
        for x, y in tqdm(train_loader, desc=f"{args.mode} {epoch+1}/{args.epochs}"):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None:
                with torch.no_grad(), autocast():
                    t_logits, t_feat = teacher_forward_features(teacher, x)
            with autocast():
                s_logits, s_feat = student(x[:, STUDENT_LEADS])
                hard = F.cross_entropy(s_logits, y, weight=weights, label_smoothing=0.05)
                if teacher is None:
                    loss = hard
                else:
                    soft = (args.temperature ** 2) * F.kl_div(
                        F.log_softmax(s_logits / args.temperature, 1),
                        F.softmax(t_logits / args.temperature, 1), reduction="batchmean")
                    feature = F.mse_loss(s_feat, t_feat)
                    loss = hard + args.alpha * soft + args.beta * feature
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); ema.update(student)
            total += loss.item()
        scheduler.step()
        raw = deepcopy(student.state_dict()); ema.copy_to(student)
        metrics = evaluate(student, val_loader, device, autocast, teacher_mod, tta=2)
        student.load_state_dict(raw)
        history.append({"epoch": epoch+1, "train_loss": total/len(train_loader),
                        **{f"val_{k}": v for k, v in metrics.items()
                           if k not in {"probabilities", "targets"}}})
        print(f"Epoch {epoch+1}: loss={total/len(train_loader):.4f}, "
              f"val_acc={metrics['accuracy']:.2f}, val_auc={metrics['auc']:.2f}")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in ema.state.items()}
            stale = 0
        else:
            stale += 1
        torch.save({"epoch": epoch, "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "ema_state_dict": {k: v.detach().cpu() for k, v in ema.state.items()},
                    "best_state": best_state, "best_acc": best_acc,
                    "stale": stale, "history": history}, snapshot_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if stale >= args.patience: break

    student.load_state_dict(best_state)
    final_val = evaluate(student, val_loader, device, autocast, teacher_mod, tta=4)
    tuned = teacher_mod.tune_threshold(final_val["probabilities"], final_val["targets"])
    metadata = {
        "task": "MI_OR_ISCHEMIA_vs_OTHERS", "mode": args.mode,
        "leads": ["II", "aVF"], "lead_indices": STUDENT_LEADS,
        "parameters": n_params, "epochs_completed": len(history),
        "best_val_accuracy": best_acc, "threshold": tuned["threshold"],
        "validation_accuracy_at_threshold": tuned["accuracy"],
        "temperature": args.temperature if teacher is not None else None,
        "alpha": args.alpha if teacher is not None else None,
        "beta": args.beta if teacher is not None else None,
        "elapsed_seconds": time.time() - started, "test_evaluated": False,
    }
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    name = "student_mi_kd_best.pt" if args.mode == "kd" else "student_baseline_best.pt"
    torch.save({"model_state_dict": best_state, "threshold": tuned["threshold"],
                "metadata": metadata}, ckpt_dir / name)
    print(f"Saved {ckpt_dir / name}; TEST NOT EVALUATED")


if __name__ == "__main__":
    main()

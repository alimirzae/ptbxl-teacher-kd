import argparse
import json
import math
import os
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
import wfdb
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# PTB-XL 12-lead Teacher — MI/Ischemia vs Others
# Windows-native / C:\ptbxl
#
# IMPORTANT:
# - Train = folds 1..8
# - Validation = fold 9
# - Test = fold 10, NOT touched by this script
# - Threshold is tuned on validation only
# ============================================================

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        hidden = max(1, c // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout)
        self.short = (
            nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride),
                nn.BatchNorm1d(cout),
            )
            if (cin != cout or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        r = self.short(x)
        x = F.relu(self.b1(self.c1(x)))
        x = self.se(self.b2(self.c2(x)))
        return F.relu(x + r)

class TeacherBinary(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        self.layer2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )
        self.layer3 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool1d(50)
        enc = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 50, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)

class ECGDataset(Dataset):
    def __init__(self, csv_path, root, augment=False, limit=None):
        self.root = Path(root)
        self.augment = augment

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["class_id", "filename_hr"]).copy()
        df["class_id"] = df["class_id"].astype(int)

        valid_rows = []
        missing = 0
        for _, r in df.iterrows():
            base = self.root / str(r["filename_hr"])
            if base.with_suffix(".dat").exists() and base.with_suffix(".hea").exists():
                valid_rows.append(r)
            else:
                missing += 1

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if limit is not None:
            # Deterministic class-stratified subset for meaningful smoke tests.
            n = min(limit, len(self.df))
            if n < len(self.df) and self.df["class_id"].nunique() > 1:
                fractions = self.df["class_id"].value_counts(normalize=True)
                parts = []
                remaining = n
                for class_id in sorted(fractions.index):
                    take = max(1, int(round(n * fractions[class_id])))
                    take = min(take, remaining, int((self.df["class_id"] == class_id).sum()))
                    parts.append(
                        self.df[self.df["class_id"] == class_id].sample(
                            n=take, random_state=SEED
                        )
                    )
                    remaining -= take
                if remaining > 0:
                    used = set(pd.concat(parts).index)
                    parts.append(self.df.drop(index=list(used)).sample(n=remaining, random_state=SEED))
                self.df = pd.concat(parts).sort_index().reset_index(drop=True)
            else:
                self.df = self.df.iloc[:n].reset_index(drop=True)

        pos = int(self.df["class_id"].sum())
        neg = len(self.df) - pos
        print(
            f"  {Path(csv_path).name}: {len(self.df):,} usable | "
            f"positive={pos:,} | negative={neg:,} | missing={missing:,} | augment={augment}"
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _augment(sig):
        # Gaussian noise
        if np.random.rand() < 0.6:
            snr = np.random.uniform(12, 30)
            p = np.mean(sig ** 2) + 1e-8
            noise_std = np.sqrt(p / (10 ** (snr / 10)))
            sig = sig + np.random.normal(0, noise_std, sig.shape)

        # Baseline wander
        if np.random.rand() < 0.5:
            t = np.arange(sig.shape[0]) / 500.0
            amp = np.random.uniform(0.05, 0.25)
            freq = np.random.uniform(0.1, 0.6)
            sig = sig + (amp * np.sin(2 * np.pi * freq * t))[:, None]

        # Amplitude scaling
        if np.random.rand() < 0.5:
            sig = sig * np.random.uniform(0.8, 1.2)

        # Random lead dropout
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(1, 3)
            idx = np.random.choice(12, n_drop, replace=False)
            sig[:, idx] = 0.0

        # Time shift
        if np.random.rand() < 0.5:
            sig = np.roll(sig, np.random.randint(-500, 501), axis=0)

        # Cutout
        if np.random.rand() < 0.3:
            length = np.random.randint(250, 1001)
            start = np.random.randint(0, max(1, sig.shape[0] - length + 1))
            sig[start : start + length] = 0.0

        return sig

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        base = self.root / str(r["filename_hr"])

        try:
            sig, _ = wfdb.rdsamp(str(base))
        except Exception as e:
            raise RuntimeError(f"Failed reading ECG {base}: {e}") from e

        if sig.shape != (5000, 12):
            raise RuntimeError(f"Unexpected ECG shape {sig.shape} at {base}")

        sig = sig.astype(np.float32, copy=False)

        if self.augment:
            sig = self._augment(sig)

        # Per-lead Z-score
        mean = sig.mean(axis=0, keepdims=True)
        std = sig.std(axis=0, keepdims=True)
        sig = (sig - mean) / (std + 1e-8)

        x = torch.from_numpy(sig.T.copy()).float()
        y = torch.tensor(int(r["class_id"]), dtype=torch.long)
        return x, y

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.state[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.state, strict=True)

def make_autocast(device):
    if device.type != "cuda":
        from contextlib import nullcontext
        return lambda: nullcontext()

    try:
        return lambda: torch.amp.autocast("cuda")
    except Exception:
        return lambda: torch.cuda.amp.autocast()

def make_scaler(device):
    if device.type != "cuda":
        # Newer API supports disabled scaler; fallback for older versions.
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)

    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()

def tta_aug(x):
    x = x.clone()
    b = x.size(0)
    x = x * torch.empty(b, 1, 1, device=x.device).uniform_(0.95, 1.05)
    x = x + torch.empty(b, 1, 1, device=x.device).uniform_(0.005, 0.02) * torch.randn_like(x)
    for i in range(b):
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-150, 151, (1,), device=x.device).item())
            x[i] = torch.roll(x[i], shift, dims=1)
    return x

@torch.no_grad()
def evaluate(model, loader, device, autocast, tta=0, threshold=0.5):
    model.eval()
    probs_all, y_all = [], []

    for x, y in tqdm(loader, desc="Validation", leave=False):
        x = x.to(device, non_blocking=True)

        with autocast():
            probs = [torch.softmax(model(x), dim=1)[:, 1]]
            for _ in range(tta):
                probs.append(torch.softmax(model(tta_aug(x)), dim=1)[:, 1])

        p = torch.stack(probs, dim=0).mean(dim=0)
        probs_all.append(p.cpu().numpy())
        y_all.append(y.numpy())

    p = np.concatenate(probs_all)
    y = np.concatenate(y_all)
    pred = (p >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())

    acc = 100.0 * (pred == y).mean()
    sens = 100.0 * tp / max(1, tp + fn)
    spec = 100.0 * tn / max(1, tn + fp)
    f1 = 100.0 * (2 * tp) / max(1, 2 * tp + fp + fn)
    auc = 100.0 * roc_auc_score(y, p)

    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "auc": float(auc),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "probabilities": p,
        "targets": y,
    }

def tune_threshold(probabilities, targets):
    best = {"threshold": 0.5, "accuracy": -1.0}
    for thr in np.arange(0.20, 0.801, 0.01):
        pred = (probabilities >= thr).astype(np.int64)
        acc = 100.0 * (pred == targets).mean()
        if acc > best["accuracy"]:
            best = {"threshold": float(round(thr, 2)), "accuracy": float(acc)}
    return best

def weighted_ce(logits, targets, class_weight, label_smoothing=0.05):
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weight,
        label_smoothing=label_smoothing,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"C:\ptbxl"))
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-train-limit", type=int, default=32)
    parser.add_argument("--smoke-val-limit", type=int, default=32)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a full training run from the latest epoch snapshot")
    args = parser.parse_args()

    root = args.root
    split_dir = root / "project" / "data" / "splits"
    run_dir = root / "project" / "runs" / "teacher"
    ckpt_dir = root / "project" / "checkpoints" / "teacher"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "train.csv/val.csv not found. Run 02_create_splits.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("PTB-XL TEACHER — MI/ISCHEMIA vs OTHERS")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("WARNING: CUDA GPU not detected. Training will run on CPU and may be slow.")

    smoke = args.mode == "smoke"
    epochs = 1 if smoke else args.epochs
    train_limit = args.smoke_train_limit if smoke else None
    val_limit = args.smoke_val_limit if smoke else None
    val_tta = 0 if smoke else 2

    train_ds = ECGDataset(train_csv, root, augment=True, limit=train_limit)
    val_ds = ECGDataset(val_csv, root, augment=False, limit=val_limit)

    pin = device.type == "cuda"
    workers = 0 if smoke else max(0, args.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, 32) if smoke else args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.eval_batch_size, 16) if smoke else args.eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    # Class weights computed from the ACTUAL training subset used in this run.
    n_pos = int(train_ds.df["class_id"].sum())
    n_neg = len(train_ds) - n_pos
    class_weight = torch.tensor(
        [
            len(train_ds) / (2 * max(1, n_neg)),
            len(train_ds) / (2 * max(1, n_pos)),
        ],
        dtype=torch.float32,
        device=device,
    )
    print(f"Class weights: neg={class_weight[0].item():.4f}, pos={class_weight[1].item():.4f}")

    model = TeacherBinary(2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup = 1

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = make_scaler(device)
    autocast = make_autocast(device)
    ema = EMA(model, decay=0.999)

    best_val_acc = -1.0
    best_ema_state = None
    no_improve = 0
    history = []
    start_epoch = 0
    snapshot_path = ckpt_dir / "teacher_training_latest.pt"

    if args.resume and not smoke and snapshot_path.exists():
        snapshot = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(snapshot["model_state_dict"])
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        scheduler.load_state_dict(snapshot["scheduler_state_dict"])
        ema.state = {k: v.to(device) for k, v in snapshot["ema_state_dict"].items()}
        best_ema_state = snapshot.get("best_ema_state")
        best_val_acc = float(snapshot.get("best_val_acc", -1.0))
        no_improve = int(snapshot.get("no_improve", 0))
        history = snapshot.get("history", [])
        start_epoch = int(snapshot["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch} of {epochs}.")

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                if np.random.rand() < 0.5:
                    lam = np.random.beta(0.2, 0.2)
                    idx = torch.randperm(x.size(0), device=device)
                    logits = model(lam * x + (1 - lam) * x[idx])
                    loss = (
                        lam * weighted_ce(logits, y, class_weight)
                        + (1 - lam) * weighted_ce(logits, y[idx], class_weight)
                    )
                else:
                    logits = model(x)
                    loss = weighted_ce(logits, y, class_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Evaluate EMA weights without destroying train weights.
        raw_state = deepcopy(model.state_dict())
        ema.copy_to(model)
        metrics = evaluate(model, val_loader, device, autocast, tta=val_tta)
        model.load_state_dict(raw_state)

        avg_loss = total_loss / max(1, len(train_loader))
        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "val_accuracy": metrics["accuracy"],
            "val_sensitivity": metrics["sensitivity"],
            "val_specificity": metrics["specificity"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch+1:02d} | loss={avg_loss:.4f} | "
            f"Val Acc={metrics['accuracy']:.2f}% | "
            f"Sens={metrics['sensitivity']:.2f}% | "
            f"Spec={metrics['specificity']:.2f}% | "
            f"F1={metrics['f1']:.2f}% | AUC={metrics['auc']:.2f}%"
        )

        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            best_ema_state = {k: v.detach().cpu().clone() for k, v in ema.state.items()}
            no_improve = 0
            print("  Best validation EMA updated.")
        else:
            no_improve += 1

        if not smoke:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "ema_state_dict": {k: v.detach().cpu() for k, v in ema.state.items()},
                    "best_ema_state": best_ema_state,
                    "best_val_acc": best_val_acc,
                    "no_improve": no_improve,
                    "history": history,
                },
                snapshot_path,
            )
            pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

        if not smoke and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    elapsed = time.time() - start_time

    # Load best EMA and tune threshold on VALIDATION ONLY.
    if best_ema_state is None:
        raise RuntimeError("No best EMA state captured.")

    model.load_state_dict(best_ema_state)
    val_final = evaluate(
        model,
        val_loader,
        device,
        autocast,
        tta=0 if smoke else 4,
        threshold=0.5,
    )
    tuned = tune_threshold(val_final["probabilities"], val_final["targets"])

    print("\n" + "=" * 72)
    print("VALIDATION-ONLY FINALIZATION")
    print("=" * 72)
    print(f"Best threshold: {tuned['threshold']:.2f}")
    print(f"Validation accuracy at tuned threshold: {tuned['accuracy']:.2f}%")
    print("TEST SET HAS NOT BEEN LOADED OR EVALUATED BY THIS SCRIPT.")

    history_path = run_dir / ("smoke_history.csv" if smoke else "train_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    meta = {
        "task": "MI_OR_ISCHEMIA_vs_OTHERS",
        "positive_definition": "Any PTB-XL MI diagnostic code or explicit ischemia (ISC*) code",
        "seed": SEED,
        "mode": args.mode,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": min(args.batch_size, 32) if smoke else args.batch_size,
        "eval_batch_size": min(args.eval_batch_size, 16) if smoke else args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": 0.05,
        "ema_decay": 0.999,
        "best_val_accuracy_during_training": best_val_acc,
        "chosen_threshold_from_validation": tuned["threshold"],
        "validation_accuracy_at_chosen_threshold": tuned["accuracy"],
        "parameters": n_params,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "test_evaluated": False,
    }

    meta_path = run_dir / ("smoke_metadata.json" if smoke else "teacher_metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if smoke:
        ckpt_path = ckpt_dir / "teacher_mi_ischemia_smoke.pt"
    else:
        ckpt_path = ckpt_dir / "teacher_mi_binary_best.pt"

    torch.save(
        {
            "model_state_dict": best_ema_state,
            "threshold": tuned["threshold"],
            "metadata": meta,
        },
        ckpt_path,
    )

    print(f"Checkpoint: {ckpt_path}")
    print(f"History:    {history_path}")
    print(f"Metadata:   {meta_path}")
    print("=" * 72)

if __name__ == "__main__":
    # Required for safe DataLoader multiprocessing on Windows.
    main()

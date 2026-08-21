import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DEFAULT = Path(r"C:\ptbxl")
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ECG500Dataset(Dataset):
    def __init__(self, csv_path, root, augment=False, limit=None):
        self.root = Path(root)
        self.augment = augment
        df = pd.read_csv(csv_path).dropna(subset=["filename_hr", "class_id"]).copy()
        df["class_id"] = df["class_id"].astype(int)
        if limit is not None:
            n = min(int(limit), len(df))
            if n < len(df):
                parts = []
                for cls, g in df.groupby("class_id"):
                    take = max(1, round(n * len(g) / len(df)))
                    parts.append(g.sample(n=min(take, len(g)), random_state=SEED))
                df = pd.concat(parts).sample(frac=1, random_state=SEED).head(n)
        self.df = df.reset_index(drop=True)
        print(f"{Path(csv_path).name}: n={len(self.df)}, positive={int(self.df.class_id.sum())}, augment={augment}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        sig, _ = wfdb.rdsamp(str(self.root / str(r.filename_hr)))
        if sig.shape != (5000, 12):
            raise RuntimeError(f"Unexpected ECG shape {sig.shape} for {r.filename_hr}")
        sig = sig.astype(np.float32, copy=False)
        if self.augment:
            if np.random.rand() < 0.35:
                sig *= np.random.uniform(0.92, 1.08)
            if np.random.rand() < 0.30:
                sig = np.roll(sig, np.random.randint(-150, 151), axis=0)
            if np.random.rand() < 0.20:
                power = np.mean(sig ** 2) + 1e-8
                snr = np.random.uniform(26, 38)
                sig += np.random.normal(0, np.sqrt(power / 10 ** (snr / 10)), sig.shape)
        sig = (sig - sig.mean(0, keepdims=True)) / (sig.std(0, keepdims=True) + 1e-6)
        age = float(r.age) if "age" in r and pd.notna(r.age) else 60.0
        sex = float(r.sex) if "sex" in r and pd.notna(r.sex) else 0.5
        meta = np.array([(age - 60.0) / 18.0, sex], dtype=np.float32)
        return torch.from_numpy(sig.T.copy()), torch.from_numpy(meta), torch.tensor(int(r.class_id))


class SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1, kernel=7):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.c1 = nn.Conv1d(cin, cout, kernel, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, kernel, padding=pad, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout)
        self.short = nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)
        ) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        residual = self.short(x)
        x = F.silu(self.b1(self.c1(x)))
        x = self.se(self.b2(self.c2(x)))
        return F.silu(x + residual)


class FeatureBranch(nn.Module):
    def __init__(self, stem_kernel, widths, dilations):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, widths[0], stem_kernel, stride=2, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.SiLU(),
        )
        blocks = []
        cin = widths[0]
        for i, cout in enumerate(widths[1:]):
            blocks.append(ResidualBlock(cin, cout, stride=2, dilation=1, kernel=7))
            blocks.append(ResidualBlock(cout, cout, stride=1, dilation=dilations[min(i, len(dilations)-1)], kernel=5))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = widths[-1]

    def forward(self, x):
        return self.blocks(self.stem(x))


class MultiScaleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = FeatureBranch(15, [64, 96, 160, 256], [1, 2, 4])
        self.low = FeatureBranch(11, [48, 80, 128, 192], [1, 2, 4])
        self.meta = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Dropout(0.1), nn.Linear(32, 32), nn.SiLU())
        fused = (self.high.out_channels + self.low.out_channels) * 2 + 32
        self.head = nn.Sequential(
            nn.Linear(fused, 384), nn.SiLU(), nn.Dropout(0.35), nn.Linear(384, 96), nn.SiLU(), nn.Dropout(0.15), nn.Linear(96, 1)
        )

    @staticmethod
    def summarize(x):
        return torch.cat([x.mean(-1), x.amax(-1)], dim=1)

    def forward(self, x, meta):
        high = self.high(x)
        low_input = F.avg_pool1d(x, kernel_size=5, stride=5)
        low = self.low(low_input)
        fused = torch.cat([self.summarize(high), self.summarize(low), self.meta(meta)], dim=1)
        return self.head(fused).squeeze(1)


def score(y, p, threshold):
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(100 * (pred == y).mean()),
        "sensitivity": float(100 * tp / max(1, tp + fn)),
        "specificity": float(100 * tn / max(1, tn + fp)),
        "f1": float(100 * 2 * tp / max(1, 2 * tp + fp + fn)),
        "auc": float(100 * roc_auc_score(y, p)),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, meta, y in tqdm(loader, desc="Multiscale validation", leave=False):
        x = x.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x, meta))
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    best = max((score(y, p, t) for t in np.arange(0.05, 0.951, 0.001)), key=lambda r: (r["accuracy"], r["sensitivity"], r["f1"]))
    return best, y, p


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--smoke-train-limit", type=int, default=96)
    parser.add_argument("--smoke-val-limit", type=int, default=96)
    args = parser.parse_args()

    seed_everything()
    root = args.root
    out = root / "results" / "teacher_multiscale_500hz"
    out.mkdir(parents=True, exist_ok=True)
    split_dir = root / "project" / "data" / "splits"
    train_limit = args.smoke_train_limit if args.mode == "smoke" else None
    val_limit = args.smoke_val_limit if args.mode == "smoke" else None
    train = ECG500Dataset(split_dir / "train.csv", root, augment=True, limit=train_limit)
    val = ECG500Dataset(split_dir / "val.csv", root, augment=False, limit=val_limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)} CUDA={torch.version.cuda}")

    loader_args = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 3})
    val_loader = DataLoader(val, batch_size=args.eval_batch_size, shuffle=False, **loader_args)

    model = MultiScaleTeacher().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, (len(train) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    snap = out / "training_latest.pt"
    start_epoch, resume_batch, best_acc, stale = 0, 0, -1.0, 0
    history = []
    if args.mode == "train" and snap.exists():
        s = torch.load(snap, map_location=device)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        sched.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        start_epoch = int(s["epoch"])
        resume_batch = int(s.get("next_batch", 0))
        best_acc = float(s.get("best_accuracy", -1.0))
        stale = int(s.get("stale", 0))
        history = s.get("history", [])
        if resume_batch >= steps:
            start_epoch += 1
            resume_batch = 0
        print(f"RESUME epoch={start_epoch+1}/{args.epochs}, batch={resume_batch}/{steps}")

    last_save = time.time()
    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator().manual_seed(SEED + epoch)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_args)
        model.train()
        total_loss, done = 0.0, 0
        for bi, (x, meta, y) in enumerate(tqdm(train_loader, desc=f"MultiScale {epoch+1}/{args.epochs}")):
            if epoch == start_epoch and bi < resume_batch:
                continue
            x = x.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, meta)
                loss = F.binary_cross_entropy_with_logits(logits, y, label_smoothing=0.01)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
            done += 1

            if args.mode == "train" and time.time() - last_save >= 300:
                atomic_save({
                    "epoch": epoch, "next_batch": bi + 1, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "best_accuracy": best_acc,
                    "stale": stale, "history": history,
                }, snap)
                (out / "progress.json").write_text(json.dumps({
                    "scope": "validation_only", "test_evaluated": False, "epoch": epoch + 1, "epochs": args.epochs,
                    "batch": bi + 1, "batches": steps, "best_validation_accuracy": best_acc,
                }, indent=2), encoding="utf-8")
                print(f"5-minute checkpoint: epoch {epoch+1}, batch {bi+1}")
                last_save = time.time()

        metrics, yv, pv = validate(model, val_loader, device)
        row = {"epoch": epoch + 1, "mean_loss": total_loss / max(1, done), "lr": opt.param_groups[0]["lr"], **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "metrics": metrics,
                "architecture": "MultiScaleTeacher500Hz", "test_evaluated": False,
            }, out / "teacher_multiscale_best.pt")
            pd.DataFrame({"target": yv, "probability": pv}).to_csv(out / "best_validation_probabilities.csv", index=False)
        else:
            stale += 1

        if args.mode == "train":
            atomic_save({
                "epoch": epoch, "next_batch": steps, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "best_accuracy": best_acc,
                "stale": stale, "history": history,
            }, snap)
        print(json.dumps(row))
        resume_batch = 0
        if args.mode == "smoke":
            break
        if best_acc >= 90.0:
            print("TARGET REACHED ON VALIDATION")
            break
        if epoch >= 9 and stale >= args.patience:
            print("EARLY STOP: validation accuracy plateau")
            break

    report = {
        "scope": "validation_only", "test_evaluated": False, "architecture": "MultiScaleTeacher500Hz",
        "best_validation_accuracy": best_acc, "target_reached": best_acc >= 90.0, "mode": args.mode, "history": history,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("TEST SET NOT LOADED OR EVALUATED")


if __name__ == "__main__":
    main()

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
from torch.utils.data import DataLoader, Dataset, Subset

SEED = 42
ROOT_DEFAULT = Path(r"C:\ptbxl")
BENCHMARK_ACCURACY = 87.7691


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
        self.epoch = 0
        df = pd.read_csv(csv_path).dropna(subset=["filename_hr", "class_id"]).copy()
        df["class_id"] = df["class_id"].astype(int)
        if limit is not None and limit < len(df):
            n = int(limit)
            parts = []
            for _, group in df.groupby("class_id"):
                take = max(1, round(n * len(group) / len(df)))
                parts.append(group.sample(n=min(take, len(group)), random_state=SEED))
            df = pd.concat(parts).sample(frac=1, random_state=SEED).head(n)
        self.df = df.reset_index(drop=True)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig, _ = wfdb.rdsamp(str(self.root / str(row.filename_hr)))
        if sig.shape != (5000, 12):
            raise RuntimeError(f"Unexpected ECG shape {sig.shape} for {row.filename_hr}")
        sig = sig.astype(np.float32, copy=False)
        if self.augment:
            rng = np.random.default_rng(SEED + self.epoch * 100_000 + int(idx))
            if rng.random() < 0.35:
                sig *= rng.uniform(0.92, 1.08)
            if rng.random() < 0.30:
                sig = np.roll(sig, int(rng.integers(-150, 151)), axis=0)
            if rng.random() < 0.20:
                power = np.mean(sig ** 2) + 1e-8
                snr = rng.uniform(26, 38)
                sig += rng.normal(0, np.sqrt(power / 10 ** (snr / 10)), sig.shape)
        sig = (sig - sig.mean(0, keepdims=True)) / (sig.std(0, keepdims=True) + 1e-6)
        age = float(row.age) if "age" in row and pd.notna(row.age) else 60.0
        sex = float(row.sex) if "sex" in row and pd.notna(row.sex) else 0.5
        meta = np.array([(age - 60.0) / 18.0, sex], dtype=np.float32)
        return torch.from_numpy(sig.T.copy()), torch.from_numpy(meta), torch.tensor(int(row.class_id))


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(16, channels // 8)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, channels), nn.Sigmoid())

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
        self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        residual = self.short(x)
        x = F.silu(self.b1(self.c1(x)))
        x = self.se(self.b2(self.c2(x)))
        return F.silu(x + residual)


class FeatureBranch(nn.Module):
    def __init__(self, stem_kernel, widths):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, widths[0], stem_kernel, 2, stem_kernel // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.SiLU(),
        )
        blocks = []
        cin = widths[0]
        for i, cout in enumerate(widths[1:]):
            blocks += [ResidualBlock(cin, cout, 2, 1, 7), ResidualBlock(cout, cout, 1, (1, 2, 4)[i], 5)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.stem(x))


class MultiScaleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = FeatureBranch(15, [64, 96, 160, 256])
        self.low = FeatureBranch(11, [48, 80, 128, 192])
        self.meta = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Dropout(0.1), nn.Linear(32, 32), nn.SiLU())
        fused = (256 + 192) * 2 + 32
        self.head = nn.Sequential(
            nn.Linear(fused, 384), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(384, 96), nn.SiLU(), nn.Dropout(0.15), nn.Linear(96, 1),
        )

    @staticmethod
    def summarize(x):
        return torch.cat([x.mean(-1), x.amax(-1)], dim=1)

    def forward(self, x, meta):
        high = self.high(x)
        low = self.low(F.avg_pool1d(x, 5, 5))
        return self.head(torch.cat([self.summarize(high), self.summarize(low), self.meta(meta)], dim=1)).squeeze(1)


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


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_loader(dataset, indices, batch_size, workers, device):
    kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers > 0:
        kwargs.update({"persistent_workers": False, "prefetch_factor": 2})
    return DataLoader(Subset(dataset, list(map(int, indices))), **kwargs)


def save_status(out, state, run_status, args, last_validation=None):
    payload = {
        "schema_version": 1,
        "experiment": "teacher_multiscale_staged",
        "scope": "validation_only",
        "test_evaluated": False,
        "run_status": run_status,
        "mode": args.mode,
        "phase": state["phase"],
        "epoch_index_zero_based": int(state["epoch"]),
        "epoch_display": int(state["epoch"]) + 1,
        "max_epochs": int(args.epochs),
        "train_next_sample": int(state["train_next_sample"]),
        "validation_next_sample": int(state["val_next_sample"]),
        "best_validation_accuracy": float(state["best_accuracy"]),
        "benchmark_accuracy": BENCHMARK_ACCURACY,
        "benchmark_beaten": bool(state["best_accuracy"] > BENCHMARK_ACCURACY),
        "stale_epochs": int(state["stale"]),
        "last_validation": last_validation,
        "recommended_next_action": "rerun_same_stage" if run_status == "needs_resume" else ("review_results" if run_status == "completed" else "inspect_failure"),
    }
    (out / "stage_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("STAGE_STATUS_JSON=" + json.dumps(payload, separators=(",", ":")))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    ap.add_argument("--max-seconds", type=int, default=420)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--eval-batch-size", type=int, default=40)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--smoke-train-limit", type=int, default=80)
    ap.add_argument("--smoke-val-limit", type=int, default=80)
    args = ap.parse_args()
    if args.mode == "smoke":
        args.epochs = 1

    if args.max_seconds < 60 or args.max_seconds > 480:
        raise ValueError("--max-seconds must be between 60 and 480 to preserve runner timeout margin")

    seed_everything()
    root = args.root
    out = root / "results" / ("teacher_multiscale_staged_smoke" if args.mode == "smoke" else "teacher_multiscale_staged")
    out.mkdir(parents=True, exist_ok=True)
    split = root / "project" / "data" / "splits"
    train_limit = args.smoke_train_limit if args.mode == "smoke" else None
    val_limit = args.smoke_val_limit if args.mode == "smoke" else None
    train = ECG500Dataset(split / "train.csv", root, augment=True, limit=train_limit)
    val = ECG500Dataset(split / "val.csv", root, augment=False, limit=val_limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)} CUDA={torch.version.cuda}")

    model = MultiScaleTeacher().to(device)
    full_train_steps = (len(train) + args.batch_size - 1) // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=full_train_steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    snap = out / "training_latest.pt"
    state = {
        "phase": "train", "epoch": 0, "train_next_sample": 0, "val_next_sample": 0,
        "train_loss_sum": 0.0, "train_loss_count": 0, "val_targets": [], "val_probs": [],
        "best_accuracy": -1.0, "stale": 0, "history": [],
    }
    if snap.exists():
        saved = torch.load(snap, map_location=device)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state.update(saved["state"])
        print(f"RESUME phase={state['phase']} epoch={state['epoch']+1} train_sample={state['train_next_sample']} val_sample={state['val_next_sample']}")

    started = time.monotonic()
    last_checkpoint = started
    last_validation = state["history"][-1] if state["history"] else None

    def checkpoint():
        nonlocal last_checkpoint
        atomic_save({
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "state": state,
        }, snap)
        last_checkpoint = time.monotonic()

    while True:
        if state["epoch"] >= args.epochs:
            save_status(out, state, "completed", args, last_validation)
            break

        if state["phase"] == "train":
            train.set_epoch(state["epoch"])
            order = np.random.default_rng(SEED + state["epoch"]).permutation(len(train))
            remaining = order[state["train_next_sample"]:]
            loader = make_loader(train, remaining, args.batch_size, args.num_workers, device)
            model.train()
            for x, meta, y in loader:
                x = x.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True)
                y = y.float().to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                smooth_y = y * 0.99 + 0.005
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(x, meta)
                    loss = F.binary_cross_entropy_with_logits(logits, smooth_y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                state["train_loss_sum"] += float(loss.item())
                state["train_loss_count"] += 1
                state["train_next_sample"] += int(y.numel())

                now = time.monotonic()
                if now - last_checkpoint >= 120:
                    checkpoint()
                if now - started >= args.max_seconds:
                    checkpoint()
                    save_status(out, state, "needs_resume", args, last_validation)
                    return

            state["phase"] = "validate"
            state["val_next_sample"] = 0
            state["val_targets"] = []
            state["val_probs"] = []
            checkpoint()

        if state["phase"] == "validate":
            remaining = np.arange(state["val_next_sample"], len(val))
            loader = make_loader(val, remaining, args.eval_batch_size, args.num_workers, device)
            model.eval()
            with torch.no_grad():
                for x, meta, y in loader:
                    x = x.to(device, non_blocking=True)
                    meta = meta.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        p = torch.sigmoid(model(x, meta))
                    state["val_targets"].extend(y.numpy().astype(int).tolist())
                    state["val_probs"].extend(p.cpu().numpy().astype(float).tolist())
                    state["val_next_sample"] += int(y.numel())

                    now = time.monotonic()
                    if now - last_checkpoint >= 120:
                        checkpoint()
                    if now - started >= args.max_seconds:
                        checkpoint()
                        save_status(out, state, "needs_resume", args, last_validation)
                        return

            yv = np.asarray(state["val_targets"], dtype=np.int64)
            pv = np.asarray(state["val_probs"], dtype=np.float32)
            metrics = max((score(yv, pv, t) for t in np.arange(0.05, 0.951, 0.001)), key=lambda r: (r["accuracy"], r["sensitivity"], r["f1"]))
            row = {
                "epoch": int(state["epoch"]) + 1,
                "mean_loss": float(state["train_loss_sum"] / max(1, state["train_loss_count"])),
                "lr": float(opt.param_groups[0]["lr"]),
                **metrics,
            }
            state["history"].append(row)
            last_validation = row
            pd.DataFrame(state["history"]).to_csv(out / "history.csv", index=False)

            if metrics["accuracy"] > state["best_accuracy"]:
                state["best_accuracy"] = float(metrics["accuracy"])
                state["stale"] = 0
                torch.save({
                    "model_state_dict": model.state_dict(), "metrics": metrics,
                    "architecture": "MultiScaleTeacher500HzStaged", "test_evaluated": False,
                }, out / "teacher_multiscale_best.pt")
                pd.DataFrame({"target": yv, "probability": pv}).to_csv(out / "best_validation_probabilities.csv", index=False)
            else:
                state["stale"] += 1

            report = {
                "scope": "validation_only", "test_evaluated": False, "architecture": "MultiScaleTeacher500HzStaged",
                "benchmark_validation_accuracy": BENCHMARK_ACCURACY, "best_validation_accuracy": state["best_accuracy"],
                "benchmark_beaten": bool(state["best_accuracy"] > BENCHMARK_ACCURACY), "history": state["history"],
            }
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("VALIDATION_RESULT=" + json.dumps(row, separators=(",", ":")))

            state["epoch"] += 1
            state["phase"] = "train"
            state["train_next_sample"] = 0
            state["val_next_sample"] = 0
            state["train_loss_sum"] = 0.0
            state["train_loss_count"] = 0
            state["val_targets"] = []
            state["val_probs"] = []
            checkpoint()

            if args.mode == "smoke" or state["best_accuracy"] >= 90.0 or (state["epoch"] >= 10 and state["stale"] >= args.patience):
                save_status(out, state, "completed", args, last_validation)
                return

            if time.monotonic() - started >= args.max_seconds:
                save_status(out, state, "needs_resume", args, last_validation)
                return


if __name__ == "__main__":
    main()

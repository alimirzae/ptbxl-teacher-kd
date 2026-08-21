import json
import os
import random
import time
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(r"C:\ptbxl")
OUT = Path(os.environ.get(
    "PTBXL_FINETUNE_OUT",
    ROOT / "results/teacher_optimization/finetune_weak_aug",
))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def load_module():
    spec = spec_from_file_location("teacher_pipeline", ROOT / "03_train_teacher.py")
    module = module_from_spec(spec); spec.loader.exec_module(module); return module


teacher_mod = load_module()


def atomic_torch_save(payload, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp); os.replace(tmp, path)
    except Exception:
        if tmp.exists(): tmp.unlink()
        raise


class WeakAugDataset(teacher_mod.ECGDataset):
    @staticmethod
    def _augment(sig):
        if np.random.rand() < 0.35:
            snr = np.random.uniform(20, 35); power = np.mean(sig ** 2) + 1e-8
            sig = sig + np.random.normal(0, np.sqrt(power / (10 ** (snr / 10))), sig.shape)
        if np.random.rand() < 0.4:
            sig = sig * np.random.uniform(0.9, 1.1)
        if np.random.rand() < 0.3:
            sig = np.roll(sig, np.random.randint(-100, 101), axis=0)
        return sig


def metric_row(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    from sklearn.metrics import roc_auc_score
    return {"threshold": float(threshold), "accuracy": float(100*(pred == y).mean()),
            "sensitivity": float(100*tp/max(1,tp+fn)), "specificity": float(100*tn/max(1,tn+fp)),
            "f1": float(100*2*tp/max(1,2*tp+fp+fn)), "auc": float(100*roc_auc_score(y,p)),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def tune(y, p):
    rows = [metric_row(y, p, round(float(t),3)) for t in np.arange(.05,.951,.001)]
    return max(rows, key=lambda r: (r["accuracy"], r["sensitivity"], r["f1"]))


@torch.no_grad()
def validate(model, loader, device, autocast):
    model.eval(); ps=[]; ys=[]
    torch.manual_seed(SEED)
    for x,y in tqdm(loader, desc="Fine-tune validation", leave=False):
        x=x.to(device, non_blocking=True)
        with autocast():
            clean=torch.softmax(model(x),1)[:,1]
            aug=torch.softmax(model(teacher_mod.tta_aug(x)),1)[:,1]
        ps.append(((clean+aug)/2).cpu().numpy()); ys.append(y.numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    return tune(y,p)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast=teacher_mod.make_autocast(device); scaler=teacher_mod.make_scaler(device)
    train=WeakAugDataset(ROOT/"project/data/splits/train.csv",ROOT,augment=True)
    val=teacher_mod.ECGDataset(ROOT/"project/data/splits/val.csv",ROOT,augment=False)
    val_loader=DataLoader(val,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    source=ROOT/"project/checkpoints/teacher/teacher_mi_binary_best.pt"
    ckpt=torch.load(source,map_location=device)
    model=teacher_mod.TeacherBinary().to(device); model.load_state_dict(ckpt["model_state_dict"])
    n_pos=int(train.df.class_id.sum()); n_neg=len(train)-n_pos
    weights=torch.tensor([len(train)/(2*n_neg),len(train)/(2*n_pos)],dtype=torch.float32,device=device)
    optimizer=optim.AdamW(model.parameters(),lr=1e-5,weight_decay=5e-5)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-6)
    ema=teacher_mod.EMA(model,decay=.99)
    best_acc=87.44846541456711; best_state=None; best_metrics=None; history=[]
    snapshot_path=OUT/"training_latest.pt"; progress_path=OUT/"progress.json"
    start_epoch=0; resume_batch=0; resume_total=0.0; resume_rng=None
    if snapshot_path.exists():
        snapshot=torch.load(snapshot_path,map_location=device)
        model.load_state_dict(snapshot["model_state_dict"])
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        scheduler.load_state_dict(snapshot["scheduler_state_dict"])
        ema.state={k:v.to(device) for k,v in snapshot["ema_state_dict"].items()}
        history=snapshot.get("history",[]); best_acc=float(snapshot.get("best_acc",best_acc))
        if (OUT/"history.csv").exists():
            disk_history=pd.read_csv(OUT/"history.csv").to_dict(orient="records")
            if len(disk_history) > len(history): history=disk_history
        best_state=None; best_metrics=snapshot.get("best_metrics")
        start_epoch=int(snapshot["epoch"]); resume_batch=int(snapshot.get("next_batch",0))
        resume_total=float(snapshot.get("epoch_loss_total",0.0)); resume_rng=snapshot.get("rng_state")
        if history and int(history[-1]["epoch"]) == start_epoch+1:
            if float(history[-1]["accuracy"]) > best_acc:
                best_acc=float(history[-1]["accuracy"])
                best_metrics={k:history[-1][k] for k in ["threshold","accuracy","sensitivity",
                                                          "specificity","f1","auc","tp","fn","tn","fp"]}
            scheduler.step(); start_epoch += 1; resume_batch=0; resume_total=0.0; resume_rng=None
        print(f"Resuming fine-tune at epoch {start_epoch+1}/5, batch {resume_batch}/545")
    started=time.time(); checkpoint_interval=300.0
    for epoch in range(start_epoch,5):
        np.random.seed(SEED+epoch); random.seed(SEED+epoch); torch.manual_seed(SEED+epoch)
        generator=torch.Generator(); generator.manual_seed(SEED+epoch)
        train_loader=DataLoader(train,batch_size=32,shuffle=True,num_workers=0,pin_memory=True,
                                generator=generator)
        model.train(); total=resume_total if epoch==start_epoch else 0.0
        next_batch=resume_batch if epoch==start_epoch else 0
        last_checkpoint=time.time()
        pbar=tqdm(enumerate(train_loader),total=len(train_loader),desc=f"Fine-tune {epoch+1}/5")
        for batch_idx,(x,y) in pbar:
            if batch_idx < next_batch:
                continue
            if batch_idx == next_batch and resume_rng is not None and epoch==start_epoch:
                random.setstate(resume_rng["python"]); np.random.set_state(resume_rng["numpy"])
                cpu_rng = resume_rng["torch_cpu"]
                if not torch.is_tensor(cpu_rng):
                    cpu_rng = torch.as_tensor(cpu_rng, dtype=torch.uint8)
                torch.set_rng_state(cpu_rng.detach().cpu().to(dtype=torch.uint8))
                if device.type=="cuda" and resume_rng["torch_cuda"] is not None:
                    cuda_rng = [
                        (state if torch.is_tensor(state) else torch.as_tensor(state, dtype=torch.uint8))
                        .detach().cpu().to(dtype=torch.uint8)
                        for state in resume_rng["torch_cuda"]
                    ]
                    torch.cuda.set_rng_state_all(cuda_rng)
                resume_rng=None
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits=model(x)
                loss=F.cross_entropy(logits,y,weight=weights,label_smoothing=.02)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); ema.update(model); total += loss.item()
            next_batch=batch_idx+1
            if time.time()-last_checkpoint >= checkpoint_interval or next_batch==len(train_loader):
                rng={"python":random.getstate(),"numpy":np.random.get_state(),
                     "torch_cpu":torch.get_rng_state(),
                     "torch_cuda":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
                payload={"epoch":epoch,"next_batch":next_batch,"batches_per_epoch":len(train_loader),
                         "epoch_loss_total":total,"model_state_dict":model.state_dict(),
                         "optimizer_state_dict":optimizer.state_dict(),
                         "scheduler_state_dict":scheduler.state_dict(),
                         "ema_state_dict":{k:v.detach().cpu() for k,v in ema.state.items()},
                         "history":history,"best_acc":best_acc,
                         "best_metrics":best_metrics,"rng_state":rng}
                atomic_torch_save(payload,snapshot_path)
                progress={"stage":"teacher_weak_aug_finetune","epoch":epoch+1,"epochs":5,
                          "batch":next_batch,"batches_per_epoch":len(train_loader),
                          "epoch_percent":100*next_batch/len(train_loader),
                          "overall_percent":100*(epoch+next_batch/len(train_loader))/5,
                          "running_loss":total/max(1,next_batch),"best_validation_accuracy":best_acc,
                          "checkpoint":str(snapshot_path),"saved_at_unix":time.time(),
                          "test_evaluated":False}
                tmp=progress_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(progress,indent=2),encoding="utf-8"); os.replace(tmp,progress_path)
                last_checkpoint=time.time(); print(f"\n5-minute checkpoint saved at batch {next_batch}")
        scheduler.step()
        raw=deepcopy(model.state_dict()); ema.copy_to(model)
        metrics=validate(model,val_loader,device,autocast)
        candidate={k:v.detach().cpu().clone() for k,v in ema.state.items()}
        model.load_state_dict(raw)
        row={"epoch":epoch+1,"loss":total/len(train_loader),"lr":optimizer.param_groups[0]["lr"],**metrics}
        history.append(row); pd.DataFrame(history).to_csv(OUT/"history.csv",index=False)
        if metrics["accuracy"] > best_acc:
            best_acc=metrics["accuracy"]; best_state=candidate; best_metrics=metrics
            torch.save({"model_state_dict":best_state,"threshold":metrics["threshold"],
                        "metadata":{"source":str(source),"phase":"weak_aug_finetune",
                                    "epoch":epoch+1,"validation":metrics,"test_evaluated":False}},
                       OUT/"teacher_finetuned_best.pt")
        atomic_torch_save({"epoch":epoch+1,"next_batch":0,"batches_per_epoch":len(train_loader),
                    "epoch_loss_total":0.0,"model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),
                    "ema_state_dict":{k:v.detach().cpu() for k,v in ema.state.items()},
                    "history":history,"best_acc":best_acc,"best_metrics":best_metrics,
                    "rng_state":None},snapshot_path)
        print(json.dumps(row,indent=2))
        resume_batch=0; resume_total=0.0; resume_rng=None
        if best_acc > 90.0: break
    report={"source_checkpoint":str(source),"test_evaluated":False,"history":history,
            "best_validation_accuracy":best_acc,"target_reached":best_acc>90.0,
            "best_metrics":best_metrics,"elapsed_seconds":time.time()-started}
    (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); print("TEST SET NOT LOADED OR EVALUATED")


if __name__ == "__main__": main()

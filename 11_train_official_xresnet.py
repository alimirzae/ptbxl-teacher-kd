import importlib, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT=Path(r"C:\ptbxl"); OUT=ROOT/"results/teacher_official_xresnet101"; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/"external/official_ptbxl_models"))
sys.path.insert(0,str(ROOT/"external"))
from official_ptbxl_models.xresnet1d import xresnet1d101
sys.path.insert(0, str(ROOT))
base=importlib.import_module("10_train_fast_teacher_100hz")

SEED=42
def atomic(obj,path):
    tmp=Path(str(path)+".tmp"); torch.save(obj,tmp); os.replace(tmp,path)

@torch.no_grad()
def validate(model,loader,device):
    model.eval(); ys=[]; ps=[]
    for x,_,y in tqdm(loader,desc="XResNet validation",leave=False):
        x=x.to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
            p=torch.sigmoid(model(x).squeeze(1))
        ys.append(y.numpy()); ps.append(p.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    best=max((base.score(y,p,t) for t in np.arange(.05,.951,.001)),key=lambda r:(r["accuracy"],r["sensitivity"],r["f1"]))
    return best,y,p

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.backends.cudnn.benchmark=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=base.ECG100Dataset(ROOT/"project/data/splits/train.csv",augment=True)
    val=base.ECG100Dataset(ROOT/"project/data/splits/val.csv",augment=False)
    kwargs={"num_workers":4,"pin_memory":True,"persistent_workers":True,"prefetch_factor":4}
    vl=DataLoader(val,batch_size=128,shuffle=False,**kwargs)
    model=xresnet1d101(input_channels=12,num_classes=1,kernel_size=5,kernel_size_stem=5,ps_head=.35).to(device)
    epochs=20; bs=128; steps=(len(train)+bs-1)//bs
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,epochs=epochs,steps_per_epoch=steps,pct_start=.15)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    snap=OUT/"training_latest.pt"; start=0; skip=0; best=-1.; history=[]; stale=0
    if snap.exists():
        s=torch.load(snap,map_location=device); model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"]); sched.load_state_dict(s["scheduler"]); scaler.load_state_dict(s["scaler"])
        start=int(s["epoch"]); skip=int(s.get("next_batch",0)); best=float(s.get("best",-1)); history=s.get("history",[]); stale=int(s.get("stale",0))
        if skip>=steps: start+=1; skip=0
        print(f"RESUME epoch {start+1}/{epochs}, batch {skip}/{steps}")
    last=time.time()
    for epoch in range(start,epochs):
        g=torch.Generator().manual_seed(SEED+epoch); tl=DataLoader(train,batch_size=bs,shuffle=True,generator=g,**kwargs)
        model.train(); total=0.; done=0
        for bi,(x,_,y) in enumerate(tqdm(tl,desc=f"Official XResNet101 {epoch+1}/{epochs}")):
            if epoch==start and bi<skip: continue
            x=x.to(device,non_blocking=True); y=y.float().to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                loss=F.binary_cross_entropy_with_logits(model(x).squeeze(1),y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(opt); scaler.update(); sched.step(); total+=loss.item(); done+=1
            if time.time()-last>=300:
                atomic({"epoch":epoch,"next_batch":bi+1,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"best":best,"history":history,"stale":stale},snap)
                (OUT/"progress.json").write_text(json.dumps({"epoch":epoch+1,"epochs":epochs,"batch":bi+1,"batches":steps,"overall_percent":100*(epoch+(bi+1)/steps)/epochs,"best_validation_accuracy":best,"test_evaluated":False},indent=2)); print(f"5-minute checkpoint: {bi+1}"); last=time.time()
        met,yv,pv=validate(model,vl,device); row={"epoch":epoch+1,"mean_loss":total/max(1,done),"lr":opt.param_groups[0]["lr"],**met}; history.append(row); pd.DataFrame(history).to_csv(OUT/"history.csv",index=False)
        if met["accuracy"]>best:
            best=met["accuracy"]; stale=0; torch.save({"model_state_dict":model.state_dict(),"metrics":met,"source":"helme/ecg_ptbxl_benchmarking xresnet1d101"},OUT/"teacher_xresnet101_best.pt"); pd.DataFrame({"target":yv,"probability":pv}).to_csv(OUT/"best_validation_probabilities.csv",index=False)
        else: stale+=1
        atomic({"epoch":epoch,"next_batch":steps,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"best":best,"history":history,"stale":stale},snap)
        print(json.dumps(row)); skip=0
        if best>=90.: print("TARGET REACHED"); break
        if epoch>=9 and stale>=6: print("EARLY STOP: no validation accuracy improvement"); break
    report={"scope":"validation_only","test_evaluated":False,"architecture":"official_xresnet1d101_adapted_binary","best_validation_accuracy":best,"target_reached":best>=90.,"history":history}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); print("TEST SET NOT LOADED OR EVALUATED")

if __name__=="__main__": main()

import json, os, random, time
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

ROOT = Path(r"C:\ptbxl")
OUT = ROOT / "results" / "teacher_fast_100hz"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


class ECG100Dataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path).dropna(subset=["filename_lr", "class_id"]).reset_index(drop=True)
        self.augment = augment
        print(f"{Path(csv_path).name}: n={len(self.df)}, positive={int(self.df.class_id.sum())}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        sig, _ = wfdb.rdsamp(str(ROOT / str(r.filename_lr)))
        sig = sig.astype(np.float32, copy=False)
        if self.augment:
            if np.random.rand() < .35: sig *= np.random.uniform(.9, 1.1)
            if np.random.rand() < .30: sig = np.roll(sig, np.random.randint(-20, 21), axis=0)
            if np.random.rand() < .25:
                power = np.mean(sig ** 2) + 1e-8
                sig += np.random.normal(0, np.sqrt(power / 10 ** (np.random.uniform(22, 35) / 10)), sig.shape)
        sig = (sig - sig.mean(0, keepdims=True)) / (sig.std(0, keepdims=True) + 1e-6)
        age = float(r.age) if pd.notna(r.age) else 60.0
        sex = float(r.sex) if pd.notna(r.sex) else .5
        meta = np.array([(age - 60.0) / 18.0, sex], dtype=np.float32)
        return torch.from_numpy(sig.T.copy()), torch.from_numpy(meta), torch.tensor(int(r.class_id))


class SE(nn.Module):
    def __init__(self, c):
        super().__init__(); self.fc = nn.Sequential(nn.Linear(c, max(8, c//8)), nn.SiLU(), nn.Linear(max(8,c//8), c), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x.mean(-1)).unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.c1=nn.Conv1d(cin,cout,7,stride,3*dilation,dilation=dilation,bias=False); self.b1=nn.BatchNorm1d(cout)
        self.c2=nn.Conv1d(cout,cout,5,1,2*dilation,dilation=dilation,bias=False); self.b2=nn.BatchNorm1d(cout)
        self.se=SE(cout); self.short=nn.Sequential(nn.Conv1d(cin,cout,1,stride,bias=False),nn.BatchNorm1d(cout)) if cin!=cout or stride!=1 else nn.Identity()
    def forward(self,x):
        r=self.short(x); x=F.silu(self.b1(self.c1(x))); x=self.se(self.b2(self.c2(x))); return F.silu(x+r)


class FastTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(12,64,15,2,7,bias=False),nn.BatchNorm1d(64),nn.SiLU(),
            Block(64,96,2),Block(96,96),Block(96,160,2),Block(160,160,1,2),
            Block(160,256,2),Block(256,256,1,2),Block(256,384,2),Block(384,384))
        self.meta=nn.Sequential(nn.Linear(2,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(384*2+32,256),nn.SiLU(),nn.Dropout(.3),nn.Linear(256,1))
    def forward(self,x,m):
        x=self.net(x); x=torch.cat([x.mean(-1),x.amax(-1),self.meta(m)],1); return self.head(x).squeeze(1)


def score(y,p,t):
    z=(p>=t); tp=int(((z==1)&(y==1)).sum()); fn=int(((z==0)&(y==1)).sum()); tn=int(((z==0)&(y==0)).sum()); fp=int(((z==1)&(y==0)).sum())
    return {"threshold":float(t),"accuracy":float(100*(z==y).mean()),"sensitivity":float(100*tp/max(1,tp+fn)),
            "specificity":float(100*tn/max(1,tn+fp)),"f1":float(100*2*tp/max(1,2*tp+fp+fn)),
            "auc":float(100*roc_auc_score(y,p)),"tp":tp,"fn":fn,"tn":tn,"fp":fp}


@torch.no_grad()
def validate(model,loader,device):
    model.eval(); ys=[]; ps=[]
    for x,m,y in tqdm(loader,desc="Validation",leave=False):
        x=x.to(device,non_blocking=True); m=m.to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
            p=torch.sigmoid(model(x,m))
        ys.append(y.numpy()); ps.append(p.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    best=max((score(y,p,t) for t in np.arange(.05,.951,.001)),key=lambda r:(r["accuracy"],r["sensitivity"],r["f1"]))
    return best,y,p


def atomic_save(obj,path):
    tmp=Path(str(path)+".tmp"); torch.save(obj,tmp); os.replace(tmp,path)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.backends.cudnn.benchmark=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=ECG100Dataset(ROOT/"project/data/splits/train.csv",True); val=ECG100Dataset(ROOT/"project/data/splits/val.csv")
    loader_args={"num_workers":4,"pin_memory":True,"persistent_workers":True,"prefetch_factor":4}
    vl=DataLoader(val,batch_size=128,shuffle=False,**loader_args)
    model=FastTeacher().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-4)
    epochs=20; sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,epochs=epochs,steps_per_epoch=(len(train)+127)//128,pct_start=.2)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    snap=OUT/"training_latest.pt"; start_epoch=0; resume_batch=0; best_acc=-1.; history=[]
    if snap.exists():
        s=torch.load(snap,map_location=device); model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"]); sched.load_state_dict(s["scheduler"]); scaler.load_state_dict(s["scaler"])
        start_epoch=int(s["epoch"]); resume_batch=int(s.get("next_batch",0)); best_acc=float(s.get("best_accuracy",-1)); history=s.get("history",[])
        if resume_batch >= (len(train)+127)//128: start_epoch+=1; resume_batch=0
        print(f"RESUME epoch={start_epoch+1}/{epochs}, batch={resume_batch}")
    last_save=time.time()
    for epoch in range(start_epoch,epochs):
        g=torch.Generator().manual_seed(SEED+epoch)
        tl=DataLoader(train,batch_size=128,shuffle=True,generator=g,**loader_args)
        model.train(); total=0.; next_batch=resume_batch if epoch==start_epoch else 0
        for bi,(x,m,y) in enumerate(tqdm(tl,desc=f"FastTeacher {epoch+1}/{epochs}")):
            if bi<next_batch: continue
            x=x.to(device,non_blocking=True); m=m.to(device,non_blocking=True); y=y.float().to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits=model(x,m); loss=F.binary_cross_entropy_with_logits(logits,y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(opt); scaler.update(); sched.step(); total+=loss.item(); next_batch=bi+1
            if time.time()-last_save>=300:
                atomic_save({"epoch":epoch,"next_batch":next_batch,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"best_accuracy":best_acc,"history":history},snap)
                (OUT/"progress.json").write_text(json.dumps({"epoch":epoch+1,"epochs":epochs,"batch":next_batch,"batches":len(tl),"overall_percent":100*(epoch+next_batch/len(tl))/epochs,"best_validation_accuracy":best_acc,"test_evaluated":False},indent=2))
                print(f"5-minute checkpoint: epoch {epoch+1}, batch {next_batch}"); last_save=time.time()
        met,yv,pv=validate(model,vl,device); row={"epoch":epoch+1,"loss":total/len(tl),"lr":opt.param_groups[0]["lr"],**met}; history.append(row)
        pd.DataFrame(history).to_csv(OUT/"history.csv",index=False)
        if met["accuracy"]>best_acc:
            best_acc=met["accuracy"]; torch.save({"model_state_dict":model.state_dict(),"metrics":met,"architecture":"FastTeacher100Hz+age+sex"},OUT/"teacher_fast_best.pt")
            pd.DataFrame({"target":yv,"probability":pv}).to_csv(OUT/"best_validation_probabilities.csv",index=False)
        atomic_save({"epoch":epoch,"next_batch":len(tl),"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"best_accuracy":best_acc,"history":history},snap)
        print(json.dumps(row)); resume_batch=0
        if best_acc>=90.: print("TARGET REACHED ON VALIDATION"); break
    report={"scope":"validation_only","test_evaluated":False,"best_validation_accuracy":best_acc,"target_reached":best_acc>=90.,"history":history}
    (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); print("TEST SET NOT LOADED OR EVALUATED")


if __name__=="__main__": main()

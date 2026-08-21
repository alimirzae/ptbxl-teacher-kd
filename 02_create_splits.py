from pathlib import Path
import ast
import pandas as pd

ROOT = Path(r"C:\ptbxl")
OUT = ROOT / "project" / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "ptbxl_database.csv")
scp = pd.read_csv(ROOT / "scp_statements.csv", index_col=0)
df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)

MI_CODES = set(scp.index[scp["diagnostic_class"].eq("MI")])
ISCHEMIA_CODES = {
    code for code, row in scp.iterrows()
    if str(code).startswith("ISC") or "ischemi" in str(row.get("description", "")).lower()
}
POSITIVE_CODES = MI_CODES | ISCHEMIA_CODES

def binary_label(d):
    codes = set(d)
    return int(bool(codes.intersection(POSITIVE_CODES)))

def clinical_label(d):
    codes = set(d)
    has_mi = bool(codes.intersection(MI_CODES))
    has_ischemia = bool(codes.intersection(ISCHEMIA_CODES))
    if has_mi and has_ischemia:
        return "MI_AND_ISCHEMIA"
    if has_mi:
        return "MI"
    if has_ischemia:
        return "ISCHEMIA"
    return "OTHER"

df["class_id"] = df["scp_dict"].apply(binary_label)
df["label"] = df["scp_dict"].apply(clinical_label)

splits = {
    "train": df[df["strat_fold"].between(1, 8)],
    "val": df[df["strat_fold"] == 9],
    "test": df[df["strat_fold"] == 10],
}

keep = [
    "ecg_id", "patient_id", "age", "sex", "strat_fold",
    "filename_lr", "filename_hr", "scp_codes", "label", "class_id"
]

for name, x in splits.items():
    x = x[keep].copy()
    path = OUT / f"{name}.csv"
    x.to_csv(path, index=False)
    p = int(x["class_id"].sum())
    print(
        f"{name}.csv -> {path} | total={len(x)} | "
        f"positive={p} | negative={len(x)-p}"
    )

# Integrity checks
train_patients = set(splits["train"]["patient_id"])
val_patients = set(splits["val"]["patient_id"])
test_patients = set(splits["test"]["patient_id"])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

print("Patient-disjoint integrity: PASS")
print(f"Positive definition: {len(MI_CODES)} MI codes + {len(ISCHEMIA_CODES)} ischemia codes")

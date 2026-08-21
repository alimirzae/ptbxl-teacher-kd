from pathlib import Path
import ast
import pandas as pd

ROOT = Path(r"C:\ptbxl")
DB = ROOT / "ptbxl_database.csv"
SCP = ROOT / "scp_statements.csv"

df = pd.read_csv(DB)
scp = pd.read_csv(SCP, index_col=0)
df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)

MI_CODES = set(scp.index[scp["diagnostic_class"].eq("MI")])
ISCHEMIA_CODES = {
    code for code, row in scp.iterrows()
    if str(code).startswith("ISC") or "ischemi" in str(row.get("description", "")).lower()
}
POSITIVE_CODES = MI_CODES | ISCHEMIA_CODES

def has_any(code_dict, codes):
    return bool(set(code_dict).intersection(codes))

df["heart_attack_spectrum_positive"] = df["scp_dict"].apply(
    lambda d: has_any(d, POSITIVE_CODES)
)

def split_name(fold):
    if fold <= 8:
        return "train"
    if fold == 9:
        return "val"
    if fold == 10:
        return "test"
    raise ValueError(f"Unexpected strat_fold={fold}")

df["split"] = df["strat_fold"].apply(split_name)

print("\n=== PTB-XL AUDIT ===")
print(f"ECGs: {len(df):,}")
print(f"Patients: {df['patient_id'].nunique():,}")

print("\n=== SPLITS ===")
for split in ["train", "val", "test"]:
    x = df[df["split"] == split]
    pos = int(x["heart_attack_spectrum_positive"].sum())
    neg = len(x) - pos
    print(
        f"{split:5s}: total={len(x):5d} | "
        f"positive={pos:4d} ({100*pos/len(x):.2f}%) | negative={neg:5d}"
    )

print("\n=== MI + ISCHEMIA POSITIVE CODE COUNTS ===")
for code in sorted(POSITIVE_CODES):
    n = int(df["scp_dict"].apply(lambda d: code in d).sum())
    desc = scp.loc[code, "description"] if code in scp.index else ""
    print(f"{code:6s} {n:5d}  {desc}")

# Verify patient-disjoint splits
patient_sets = {
    s: set(df.loc[df["split"] == s, "patient_id"])
    for s in ["train", "val", "test"]
}
assert patient_sets["train"].isdisjoint(patient_sets["val"])
assert patient_sets["train"].isdisjoint(patient_sets["test"])
assert patient_sets["val"].isdisjoint(patient_sets["test"])
print("\nPatient-disjoint check: PASS")

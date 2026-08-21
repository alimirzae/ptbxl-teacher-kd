"""Fast preflight for the staged MultiScale Teacher runner.

This intentionally does not train or touch fold 10. It checks only local
requirements before dispatching the bounded smoke/train workflow.
"""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(r"C:\ptbxl")


def main():
    checks = {
        "root_exists": ROOT.exists(),
        "train_split_exists": (ROOT / "project" / "data" / "splits" / "train.csv").exists(),
        "val_split_exists": (ROOT / "project" / "data" / "splits" / "val.csv").exists(),
        "cuda_available": torch.cuda.is_available(),
        "fold_10_loaded": False,
    }
    if checks["cuda_available"]:
        checks["gpu"] = torch.cuda.get_device_name(0)
        checks["cuda"] = torch.version.cuda
    checks["ready"] = all(checks.values())

    out = ROOT / "results" / "teacher_multiscale_staged"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preflight.json"
    path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))

    if not checks["ready"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

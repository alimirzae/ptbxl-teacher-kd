"""Run a bounded CUDA workload and publish machine-readable GitHub Actions proof."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def append_line(path_value: str | None, line: str) -> None:
    if path_value:
        with Path(path_value).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gpu_runner_probe.json"))
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    result: dict[str, object] = {
        "schema_version": 1,
        "started_at_utc": started.isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_duration_seconds": args.duration,
        "matrix_size": args.matrix_size,
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "sha": os.getenv("GITHUB_SHA"),
            "runner_name": os.getenv("RUNNER_NAME"),
        },
    }

    status = "failed"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access an NVIDIA CUDA device")

        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": props.name,
                "cuda_runtime": torch.version.cuda,
                "compute_capability": f"{props.major}.{props.minor}",
                "gpu_memory_bytes": props.total_memory,
            }
        )

        torch.manual_seed(42)
        left = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=torch.float16)
        right = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=torch.float16)
        for _ in range(3):
            product = left @ right
        torch.cuda.synchronize()

        iterations = 0
        start_clock = time.perf_counter()
        while time.perf_counter() - start_clock < args.duration:
            product = left @ right
            iterations += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_clock
        checksum = float(product[0, 0].float().cpu())
        if iterations < 1 or not math.isfinite(checksum):
            raise RuntimeError("CUDA workload did not produce a finite result")

        result.update(
            {
                "status": "passed",
                "iterations": iterations,
                "elapsed_seconds": elapsed,
                "estimated_tflops": 2 * args.matrix_size**3 * iterations / elapsed / 1e12,
                "result_checksum": checksum,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        status = "passed"
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc), "finished_at_utc": datetime.now(timezone.utc).isoformat()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    append_line(os.getenv("GITHUB_OUTPUT"), f"status={status}")
    append_line(os.getenv("GITHUB_OUTPUT"), f"gpu_name={result.get('gpu_name', 'unavailable')}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    append_line(summary, "## Local NVIDIA GPU proof")
    append_line(summary, "")
    append_line(summary, f"- Status: **{status.upper()}**")
    append_line(summary, f"- Host / runner: `{result['host']}` / `{result['github']['runner_name']}`")
    append_line(summary, f"- GPU: `{result.get('gpu_name', 'unavailable')}`")
    append_line(summary, f"- CUDA / PyTorch: `{result.get('cuda_runtime', 'n/a')}` / `{result['torch']}`")
    append_line(summary, f"- Workload: `{result.get('iterations', 0)}` multiplications of {args.matrix_size}x{args.matrix_size}")
    append_line(summary, f"- Elapsed: `{result.get('elapsed_seconds', 0):.3f}` seconds")
    append_line(summary, "- Full machine-readable evidence is attached as the `gpu-runner-proof` artifact.")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

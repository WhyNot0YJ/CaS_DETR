"""Shared runtime checks and statistics for deployment benchmarks."""

import os
import subprocess
from pathlib import Path

import torch


def command_output(command):
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def ensure_gpu_idle():
    allowed_pids = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in allowed_pids:
        allowed_pids.add(parent)
        try:
            parent = int(Path(f"/proc/{parent}/stat").read_text().split()[3])
        except (FileNotFoundError, ValueError, IndexError):
            break

    output = command_output([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ])
    busy = []
    for line in output.splitlines():
        pid, _, name = line.partition(",")
        if pid.strip().isdigit() and int(pid) not in allowed_pids:
            busy.append(f"{pid.strip()} ({name.strip()})")
    if busy:
        raise RuntimeError(
            "GPU benchmark requires an idle GPU; active compute processes: "
            + ", ".join(busy)
        )


def summarize(times_ms):
    values = torch.tensor(times_ms, dtype=torch.float64)
    mean = float(values.mean())
    return {
        "mean_latency_ms": mean,
        "p50_latency_ms": float(torch.quantile(values, 0.50)),
        "p95_latency_ms": float(torch.quantile(values, 0.95)),
        "fps": 1000.0 / mean,
    }

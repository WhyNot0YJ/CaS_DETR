#!/usr/bin/env python3
"""Resume the existing YOLO-family runs from 100e to a total of 132e."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("YOLO_PYTHON", os.sys.executable)
BATCH_LOG = ROOT / "logs" / "continue_132_batch.log"


def record(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with BATCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def find_job(config: Path) -> tuple[str, Path]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    protocol = "dairv2x" if "dairv2x" in config.name else "uadetrac"
    model = str(cfg["model"]["model_name"]).removesuffix(".pt")

    if config.name.startswith("yolox"):
        prefix = "yolo_yolox_" + model.rsplit("_", 1)[-1] + "_"
        candidates = list((ROOT / "logs" / protocol).glob(prefix + "*/weights/latest_ckpt.pth"))
        entry = "train_yolox.py"
    elif config.name.startswith("fasterrcnn"):
        prefix = "yolo_fasterrcnn_resnet50_fpn_"
        # Prefer the actively updated continuation checkpoint so an interrupted
        # resume does not fall back to the original epoch-100 checkpoint.
        candidates = list((ROOT / "logs" / protocol).glob(prefix + "*/weights/last.pt"))
        if not candidates:
            candidates = list((ROOT / "logs" / protocol).glob(prefix + "*/latest_checkpoint.pth"))
        entry = "train_fasterrcnn.py"
    else:
        prefix = "yolo_v" + model[4:] + "_" if model.startswith("yolo12") else "yolo_" + model + "_"
        candidates = list(
            (ROOT / "logs" / protocol).glob(prefix + "*/weights/continuation_100e_to_132e.pt")
        )
        entry = "train.py"

    if len(candidates) != 1:
        raise RuntimeError(f"{config.name}: expected one continuation checkpoint, got {candidates}")
    return entry, candidates[0]


def main() -> int:
    configs = sorted(
        path
        for path in (ROOT / "configs").glob("*.yaml")
        if path.name.startswith(("yolov", "yolox", "fasterrcnn"))
    )
    jobs = []
    for config in configs:
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        if int(cfg["training"]["epochs"]) != 132:
            raise RuntimeError(f"{config} is not configured for 132 epochs")
        entry, checkpoint = find_job(config)
        jobs.append((entry, config, checkpoint))

    BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    record(f"BATCH_START jobs={len(jobs)}")
    env = dict(os.environ)
    env["YOLO_TRT_BENCHMARK"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    failures = 0

    for index, (entry, config, checkpoint) in enumerate(jobs, 1):
        command = [
            PYTHON,
            entry,
            "--config",
            str(config.relative_to(ROOT)),
            "--resume_from_checkpoint",
            str(checkpoint.resolve()),
            "--epochs",
            "132",
        ]
        record(f"JOB_START {index}/{len(jobs)} {config.name} checkpoint={checkpoint}")
        with BATCH_LOG.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record(f"JOB_END {index}/{len(jobs)} {config.name} returncode={result.returncode}")
        failures += result.returncode != 0

    record(f"BATCH_END failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run, validate, and atomically publish the audited hierarchical evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CAS_DIR = EXPERIMENTS_DIR / "CaS-DETR"
YOLO_DIR = EXPERIMENTS_DIR / "yolo"
MANIFEST_PATH = SCRIPT_DIR / "hierarchical_eval_manifest.yml"
PYTHON = Path(sys.executable)
CHECKPOINT_PRIORITY = ("best_stg2.pth", "best.pth", "best_stg1.pth", "last.pth")
METRICS = ("mAP_50", "mAP_5095", "AP_small_50", "AP_small_5095")
METRIC_LABELS = {
    "mAP_50": "AP50",
    "mAP_5095": "AP50:95",
    "AP_small_50": "APs50",
    "AP_small_5095": "APs50:95",
}

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.hierarchical_eval import hierarchical_eval_spec, sha256_file
from common.result_paths import update_csv_rows


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    protocol: str
    hierarchical_eval: str
    dataset: str
    framework: str
    source_kind: str
    model: str
    config: str
    checkpoint: str
    checkpoint_sha256: str
    config_sha256: str
    output_dir: str
    prediction_dir: str
    trt_images: str
    groups: tuple[str, ...]
    num_classes: int
    yolox_exp: str = ""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_config_by_name(name: str) -> Path:
    matches = sorted(CAS_DIR.joinpath("configs").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one CaS config named {name!r}, found {matches}")
    return matches[0].resolve()


def _selected_checkpoint(output_dir: Path) -> Path:
    matches = [output_dir / name for name in CHECKPOINT_PRIORITY if (output_dir / name).is_file()]
    if not matches:
        raise FileNotFoundError(f"no selected checkpoint under {output_dir}")
    return matches[0].resolve()


def _resolve_yolox_exp(config_path: Path, config: Mapping[str, Any]) -> str:
    relative = str((config.get("model") or {}).get("yolox_exp_file", ""))
    if not relative:
        return ""
    candidate = Path(relative)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    if not candidate.is_file():
        candidate = YOLO_DIR / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"YOLOX exp file not found: {relative}")
    return str(candidate.resolve())


def _make_spec(
    *,
    run_id: str,
    protocol: str,
    framework: str,
    source_kind: str,
    model: str,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    prediction_dir: Path,
    groups: Sequence[str],
    protocols: Mapping[str, Mapping[str, Any]],
    num_classes: int,
    yolox_exp: str = "",
) -> RunSpec:
    protocol_info = protocols[protocol]
    return RunSpec(
        run_id=run_id,
        protocol=protocol,
        hierarchical_eval=str(protocol_info["hierarchical_eval"]),
        dataset=str(protocol_info["dataset"]),
        framework=framework,
        source_kind=source_kind,
        model=model,
        config=str(config.resolve()),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint),
        config_sha256=sha256_file(config),
        output_dir=str(output_dir.resolve()),
        prediction_dir=str(prediction_dir.resolve()),
        trt_images=str(Path(protocol_info["trt_images"]).resolve()),
        groups=tuple(str(group) for group in groups),
        num_classes=num_classes,
        yolox_exp=yolox_exp,
    )


def resolve_manifest() -> tuple[Dict[str, Any], List[RunSpec]]:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    protocols = manifest["protocols"]
    artifact_root = REPO_ROOT / "artifacts" / "hierarchical_eval_artifacts"
    runs: List[RunSpec] = []

    for item in manifest.get("yolo_runs", []):
        protocol = str(item["protocol"])
        log_dir = (YOLO_DIR / "logs" / protocol / str(item["log_dir"])).resolve()
        config_path = log_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model_name = str((config.get("model") or {}).get("model_name", "unknown"))
        lower = model_name.lower()
        if "fasterrcnn" in lower:
            framework = "fasterrcnn"
            checkpoint = log_dir / "weights" / "best.pt"
        elif "yolox" in lower:
            framework = "yolox"
            checkpoint = next(
                path
                for path in (
                    log_dir / "weights" / "best_ckpt.pth",
                    log_dir / "best_ckpt.pth",
                )
                if path.is_file()
            )
        else:
            framework = "yolo"
            checkpoint = log_dir / "weights" / "best.pt"
        runs.append(
            _make_spec(
                run_id=log_dir.name,
                protocol=protocol,
                framework=framework,
                source_kind="yolo_log",
                model=model_name.rsplit(".", 1)[0],
                config=config_path,
                checkpoint=checkpoint,
                output_dir=log_dir,
                prediction_dir=log_dir / "hierarchical_eval",
                groups=("detector_baseline",),
                protocols=protocols,
                num_classes=8 if protocol == "dairv2x_vehicle8" else 4,
                yolox_exp=_resolve_yolox_exp(config_path, config),
            )
        )

    cas_by_output: Dict[str, RunSpec] = {}
    for item in manifest.get("cas_runs", []):
        protocol = str(item["protocol"])
        output_rel = str(item["output"])
        output_dir = (CAS_DIR / "outputs" / protocol / output_rel).resolve()
        config_path = _resolve_config_by_name(str(item["config"]))
        checkpoint = _selected_checkpoint(output_dir)
        spec = _make_spec(
            run_id=output_dir.name,
            protocol=protocol,
            framework="casdeim",
            source_kind="cas_output",
            model=config_path.stem,
            config=config_path,
            checkpoint=checkpoint,
            output_dir=output_dir,
            prediction_dir=output_dir / "hierarchical_eval",
            groups=tuple(item.get("groups", ())) + ("cas_detr",),
            protocols=protocols,
            num_classes=8 if protocol == "dairv2x_vehicle8" else 4,
        )
        runs.append(spec)
        cas_by_output[output_rel] = spec

    for section in ("baseline_runs", "root_cas_runs"):
        for item in manifest.get(section, []):
            protocol = str(item["protocol"])
            run_id = str(item["run_id"])
            config_path = (REPO_ROOT / str(item["config"])).resolve()
            checkpoint = (REPO_ROOT / str(item["checkpoint"])).resolve()
            output_dir = (artifact_root / run_id).resolve()
            runs.append(
                _make_spec(
                    run_id=run_id,
                    protocol=protocol,
                    framework=str(item["framework"]),
                    source_kind="root_checkpoint",
                    model=config_path.stem,
                    config=config_path,
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    prediction_dir=output_dir / "hierarchical_eval",
                    groups=item.get("groups", ()),
                    protocols=protocols,
                    num_classes=8 if protocol == "dairv2x_vehicle8" else 4,
                )
            )

    expected = int(manifest["expected_unique_runs"])
    if len(runs) != expected:
        raise RuntimeError(f"manifest resolved {len(runs)} runs, expected {expected}")
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("run_id values are not globally unique")
    hashes = [run.checkpoint_sha256 for run in runs]
    if len(set(hashes)) != len(hashes):
        duplicate = sorted({value for value in hashes if hashes.count(value) > 1})
        raise RuntimeError(f"scheduled checkpoint hashes are not unique: {duplicate}")

    for run in runs:
        for path in (run.config, run.checkpoint, run.trt_images):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        expected_protocol = hierarchical_eval_spec(run.hierarchical_eval)["training_protocol"]
        if expected_protocol != run.protocol:
            raise RuntimeError(f"hierarchy/protocol mismatch for {run.run_id}")

    # Anchor the manifest to the selected traditional-detector source reports.
    # A replacement may name its old report row through ``supersedes`` so the
    # same manifest remains valid immediately before and after publication.
    for protocol in protocols:
        report = EXPERIMENTS_DIR / "reports" / protocol / "eval_metrics.csv"
        current_ids = {
            row.get("run_id", "")
            for row in _read_csv(report)
            if row.get("framework") in {"yolo", "yolox", "fasterrcnn"}
        }
        accepted_groups = []
        for item in manifest.get("yolo_runs", []):
            if str(item["protocol"]) != protocol:
                continue
            choices = {str(item["log_dir"])}
            if item.get("supersedes"):
                choices.add(str(item["supersedes"]))
            accepted_groups.append(choices)
        accepted_ids = set().union(*accepted_groups) if accepted_groups else set()
        valid = (
            len(current_ids) == len(accepted_groups)
            and not (current_ids - accepted_ids)
            and all(len(current_ids & choices) == 1 for choices in accepted_groups)
        )
        if not valid:
            raise RuntimeError(
                f"{protocol} source report run ids differ from manifest: "
                f"current={sorted(current_ids)}, "
                f"accepted={sorted(accepted_ids)}"
            )

    for alias in manifest.get("root_cas_aliases", []):
        alias_checkpoint = (REPO_ROOT / str(alias["checkpoint"])).resolve()
        target = cas_by_output.get(str(alias["alias_of"]))
        if target is None:
            raise KeyError(f"unknown alias target: {alias['alias_of']}")
        if sha256_file(alias_checkpoint) != target.checkpoint_sha256:
            raise RuntimeError(
                f"root alias hash mismatch: {alias_checkpoint.name} != {target.run_id}"
            )
    return manifest, runs


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _backup_reports(timestamp: str) -> Path:
    backup_root = REPO_ROOT / "artifacts" / "hierarchical_eval_backup" / timestamp
    backup_root.mkdir(parents=True, exist_ok=False)
    for protocol in ("dairv2x_vehicle8", "uadetrac_vehicle4"):
        source = EXPERIMENTS_DIR / "reports" / protocol
        shutil.copytree(source, backup_root / protocol)
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (backup_root / "git_status.txt").write_text(status, encoding="utf-8")
    return backup_root


def _new_work_dir(timestamp: str, manifest: Mapping[str, Any], runs: Sequence[RunSpec]) -> Path:
    root = REPO_ROOT / "artifacts" / "hierarchical_eval_runs" / timestamp
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "reports").mkdir()
    (root / "manifest_snapshot.yml").write_text(
        yaml.safe_dump(dict(manifest), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (root / "resolved_runs.json").write_text(
        json.dumps([asdict(run) for run in runs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _write_current_manifest_snapshot(
    work_dir: Path, manifest: Mapping[str, Any], runs: Sequence[RunSpec]
) -> None:
    """Record the manifest/config resolution used by the current invocation."""
    (work_dir / "manifest_resolved_latest.yml").write_text(
        yaml.safe_dump(dict(manifest), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (work_dir / "resolved_runs_latest.json").write_text(
        json.dumps([asdict(run) for run in runs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def prune_superseded_rows(work_dir: Path, runs: Sequence[RunSpec]) -> None:
    """Remove staged rows that no longer belong to the resolved manifest."""
    for protocol in ("dairv2x_vehicle8", "uadetrac_vehicle4"):
        allowed = {run.run_id for run in runs if run.protocol == protocol}
        for name in (
            "eval_metrics.csv",
            "fine_grained_eval_metrics.csv",
            "benchmark.csv",
        ):
            path = _stage_dir(work_dir, protocol) / name
            if not path.is_file() or not path.stat().st_size:
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            kept = [row for row in rows if row.get("run_id", "") in allowed]
            if len(kept) == len(rows):
                continue
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fields,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(kept)
            print(
                f"[PRUNE] {protocol}/{name}: removed {len(rows) - len(kept)} stale rows",
                flush=True,
            )


def _run_command(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {log.stem}", flush=True)
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit={result.returncode}: {' '.join(command)}\n"
            f"--- {log} tail ---\n{_tail(log)}"
        )
    print(f"[OK] {log.stem} ({time.time() - started:.1f}s)", flush=True)


def _stage_dir(work_dir: Path, protocol: str) -> Path:
    path = work_dir / "reports" / protocol
    path.mkdir(parents=True, exist_ok=True)
    return path


def _has_rows(path: Path, run: RunSpec, key: str, values: Iterable[str]) -> bool:
    wanted = set(values)
    found = {
        row.get(key, "")
        for row in _read_csv(path)
        if row.get("run_id") == run.run_id
        and row.get("checkpoint_sha256") == run.checkpoint_sha256
    }
    return found == wanted


def accuracy_complete(work_dir: Path, run: RunSpec) -> bool:
    stage = _stage_dir(work_dir, run.protocol)
    main_rows = [
        row for row in _read_csv(stage / "eval_metrics.csv")
        if row.get("run_id") == run.run_id
        and row.get("checkpoint_sha256") == run.checkpoint_sha256
    ]
    fine_rows = [
        row for row in _read_csv(stage / "fine_grained_eval_metrics.csv")
        if row.get("run_id") == run.run_id
        and row.get("checkpoint_sha256") == run.checkpoint_sha256
    ]
    return (
        {row.get("eval_split") for row in main_rows} == {"val", "test"}
        and {row.get("postprocess") for row in main_rows}
        == {"native_label_collapse_only"}
        and {row.get("eval_split") for row in fine_rows} == {"val", "test"}
        and {row.get("postprocess") for row in fine_rows} == {"native"}
    )


def missing_accuracy_splits(work_dir: Path, run: RunSpec) -> List[str]:
    stage = _stage_dir(work_dir, run.protocol)
    main_splits = {
        row.get("eval_split", "")
        for row in _read_csv(stage / "eval_metrics.csv")
        if row.get("run_id") == run.run_id
        and row.get("checkpoint_sha256") == run.checkpoint_sha256
        and row.get("postprocess") == "native_label_collapse_only"
    }
    fine_splits = {
        row.get("eval_split", "")
        for row in _read_csv(stage / "fine_grained_eval_metrics.csv")
        if row.get("run_id") == run.run_id
        and row.get("checkpoint_sha256") == run.checkpoint_sha256
        and row.get("postprocess") == "native"
    }
    complete = main_splits & fine_splits
    return [split for split in ("val", "test") if split not in complete]


def benchmark_complete(work_dir: Path, run: RunSpec) -> bool:
    return _has_rows(
        _stage_dir(work_dir, run.protocol) / "benchmark.csv",
        run,
        "mode",
        ("model", "end-to-end"),
    )


def sync_benchmark_speeds(work_dir: Path, run: RunSpec) -> None:
    rows = {
        row.get("mode", ""): row
        for row in _read_csv(_stage_dir(work_dir, run.protocol) / "benchmark.csv")
        if row.get("run_id") == run.run_id
        and row.get("checkpoint_sha256") == run.checkpoint_sha256
    }
    model = rows.get("model")
    end_to_end = rows.get("end-to-end")
    if not model or not end_to_end:
        raise RuntimeError(f"missing TensorRT rows while syncing: {run.run_id}")
    updated = update_csv_rows(
        _stage_dir(work_dir, run.protocol) / "eval_metrics.csv",
        match={"run_id": run.run_id},
        updates={
            "Inference_FPS": f"{float(model['fps']):.2f}",
            "Inference_Latency_ms": f"{float(model['mean_latency_ms']):.2f}",
            "EndToEnd_FPS": f"{float(end_to_end['fps']):.2f}",
            "EndToEnd_Latency_ms": f"{float(end_to_end['mean_latency_ms']):.2f}",
        },
    )
    if updated != 2:
        raise RuntimeError(
            f"expected two eval rows while syncing {run.run_id}, updated={updated}"
        )


def _base_env(work_dir: Path, run: RunSpec) -> Dict[str, str]:
    env = dict(os.environ)
    env["EXPERIMENT_RESULTS_DIR"] = str(_stage_dir(work_dir, run.protocol))
    env["EXPERIMENT_DATASET_PROTOCOL"] = run.protocol
    env.setdefault("CAS_EVAL_BATCH_SIZE", "16")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run_accuracy(work_dir: Path, run: RunSpec, *, force: bool = False) -> None:
    if accuracy_complete(work_dir, run) and not force:
        print(f"[SKIP][accuracy] {run.run_id}", flush=True)
        return
    stage = _stage_dir(work_dir, run.protocol)
    if run.source_kind == "yolo_log":
        command = [
            str(PYTHON),
            "eval_best_model.py",
            "--log_dir", run.output_dir,
            "--device", "cuda",
            "--dataset_registry", "configs/datasets.yaml",
            "--hierarchical-eval", run.hierarchical_eval,
            "--skip-pytorch-benchmark",
        ]
        cwd = YOLO_DIR
    else:
        protocol_flag = (
            "--dairv2x-vehicle8"
            if run.protocol == "dairv2x_vehicle8"
            else "--uadetrac-vehicle4"
        )
        Path(run.prediction_dir).mkdir(parents=True, exist_ok=True)
        missing_splits = missing_accuracy_splits(work_dir, run)
        command = [
            str(PYTHON),
            str(SCRIPT_DIR / "eval_deim_dfine.py"),
            "--framework", run.framework,
            "--config", run.config,
            "--resume", run.checkpoint,
            "--model-name", run.model,
            "--run-id", run.run_id,
            "--dataset-name", run.dataset,
            "--output-csv", str(stage / "eval_metrics.csv"),
            "--fine-grained-output-csv", str(stage / "fine_grained_eval_metrics.csv"),
            "--hierarchical-eval", run.hierarchical_eval,
            protocol_flag,
            "--device", "cuda",
            "--splits", ",".join(missing_splits),
            "--eval-num-workers", "2",
            "--predictions-dir", run.prediction_dir,
            "--skip-pytorch-benchmark",
        ]
        cwd = EXPERIMENTS_DIR
    _run_command(
        command,
        cwd=cwd,
        env=_base_env(work_dir, run),
        log=work_dir / "logs" / f"accuracy_{run.run_id}.log",
    )
    if not accuracy_complete(work_dir, run):
        raise RuntimeError(f"accuracy rows incomplete after successful command: {run.run_id}")


def _taxonomy_args(run: RunSpec) -> List[str]:
    evaluation_taxonomy = hierarchical_eval_spec(run.hierarchical_eval)["evaluation_taxonomy"]
    return [
        "--training-taxonomy", run.protocol,
        "--evaluation-taxonomy", str(evaluation_taxonomy),
        "--postprocess", "native_label_collapse_only",
    ]


def run_benchmark(work_dir: Path, run: RunSpec) -> None:
    if benchmark_complete(work_dir, run):
        sync_benchmark_speeds(work_dir, run)
        print(f"[SKIP][TensorRT] {run.run_id}", flush=True)
        return
    common = [
        "--config", run.config,
        "--run-id", run.run_id,
        "--model", run.model,
        "--images", run.trt_images,
        "--warmup", "100",
        "--iterations", "1000",
        *_taxonomy_args(run),
    ]
    if run.source_kind != "yolo_log":
        command = [
            str(PYTHON),
            str(CAS_DIR / "tools" / "benchmark" / "benchmark_experiment_trt.py"),
            "--framework", run.framework,
            "--checkpoint", run.checkpoint,
            "--output-dir", run.output_dir,
            "--dataset-protocol", run.protocol,
            "--builder", "python",
            *common,
        ]
        cwd = EXPERIMENTS_DIR
    elif run.framework == "fasterrcnn":
        command = [
            str(PYTHON),
            str(YOLO_DIR / "tools" / "benchmark_fasterrcnn_trt.py"),
            "--weights", run.checkpoint,
            "--output-dir", run.output_dir,
            "--num-classes", str(run.num_classes),
            "--dataset", run.dataset,
            *common,
        ]
        cwd = YOLO_DIR
    else:
        command = [
            str(PYTHON),
            str(YOLO_DIR / "tools" / "benchmark_trt.py"),
            "--weights", run.checkpoint,
            "--output-dir", run.output_dir,
            "--dataset", run.dataset,
            "--imgsz", "640",
            "--builder", "python",
            *common,
        ]
        if run.yolox_exp:
            command.extend(
                ("--yolox-exp", run.yolox_exp, "--num-classes", str(run.num_classes))
            )
        cwd = YOLO_DIR
    _run_command(
        command,
        cwd=cwd,
        env=_base_env(work_dir, run),
        log=work_dir / "logs" / f"tensorrt_{run.run_id}.log",
    )
    if not benchmark_complete(work_dir, run):
        raise RuntimeError(f"TensorRT rows incomplete after successful command: {run.run_id}")
    sync_benchmark_speeds(work_dir, run)


def _expected_by_protocol(runs: Sequence[RunSpec]) -> Dict[str, int]:
    return {
        protocol: sum(run.protocol == protocol for run in runs)
        for protocol in ("dairv2x_vehicle8", "uadetrac_vehicle4")
    }


def validate_results(work_dir: Path, runs: Sequence[RunSpec]) -> None:
    by_id = {run.run_id: run for run in runs}
    expected = _expected_by_protocol(runs)
    for protocol, run_count in expected.items():
        stage = _stage_dir(work_dir, protocol)
        main_rows = _read_csv(stage / "eval_metrics.csv")
        fine_rows = _read_csv(stage / "fine_grained_eval_metrics.csv")
        benchmark_rows = _read_csv(stage / "benchmark.csv")
        for label, rows, count in (
            ("main", main_rows, 2 * run_count),
            ("fine", fine_rows, 2 * run_count),
            ("benchmark", benchmark_rows, 2 * run_count),
        ):
            if len(rows) != count:
                raise RuntimeError(
                    f"{protocol} {label} row count={len(rows)}, expected={count}"
                )
        for row in main_rows:
            run = by_id.get(row.get("run_id", ""))
            if run is None or run.protocol != protocol:
                raise RuntimeError(f"unknown run row in {protocol}: {row.get('run_id')}")
            if row.get("checkpoint_sha256") != run.checkpoint_sha256:
                raise RuntimeError(f"checkpoint hash mismatch in main row: {run.run_id}")
            if row.get("training_taxonomy") != run.protocol:
                raise RuntimeError(f"training taxonomy mismatch: {run.run_id}")
            if row.get("postprocess") != "native_label_collapse_only":
                raise RuntimeError(f"postprocess mismatch: {run.run_id}")
            prediction = Path(row.get("prediction_file", ""))
            if not prediction.is_file():
                raise FileNotFoundError(f"missing collapsed predictions: {prediction}")
            for metric in METRICS:
                value = float(row[metric])
                if not 0.0 <= value <= 1.0:
                    raise RuntimeError(f"metric out of range: {run.run_id} {metric}={value}")
            for speed in (
                "Inference_FPS", "Inference_Latency_ms",
                "EndToEnd_FPS", "EndToEnd_Latency_ms",
            ):
                if float(row[speed]) <= 0:
                    raise RuntimeError(f"missing TensorRT speed: {run.run_id} {speed}")
        for row in fine_rows:
            run = by_id[row["run_id"]]
            if row.get("checkpoint_sha256") != run.checkpoint_sha256:
                raise RuntimeError(f"checkpoint hash mismatch in fine row: {run.run_id}")
            if not Path(row.get("prediction_file", "")).is_file():
                raise FileNotFoundError(f"missing raw predictions: {row.get('prediction_file')}")
        for row in benchmark_rows:
            run = by_id[row["run_id"]]
            if row.get("checkpoint_sha256") != run.checkpoint_sha256:
                raise RuntimeError(f"checkpoint hash mismatch in benchmark: {run.run_id}")
            if row.get("config_sha256") != run.config_sha256:
                raise RuntimeError(f"config hash mismatch in benchmark: {run.run_id}")
            engine = Path(row.get("engine", ""))
            if not engine.is_file() or not engine.with_suffix(engine.suffix + ".provenance.json").is_file():
                raise FileNotFoundError(f"missing hash-provenanced engine: {engine}")


def _best(rows: Sequence[Mapping[str, str]], metric: str) -> Mapping[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row[metric]))


def conclusion_markdown(protocol: str, rows: Sequence[Mapping[str, str]], runs: Sequence[RunSpec]) -> str:
    run_by_id = {run.run_id: run for run in runs if run.protocol == protocol}
    test_rows = [row for row in rows if row.get("eval_split") == "test"]
    lines = [
        f"# {protocol} 层级合并评测结论",
        "",
        "结论仅使用 test 原始指标；未构造综合分数。所有结果均为原生后处理后的 label-collapse-only COCOeval。",
        "",
    ]

    def add_winners(title: str, selected: Sequence[Mapping[str, str]]) -> None:
        lines.extend((f"## {title}", ""))
        if not selected:
            lines.extend(("该数据集无对应实验。", ""))
            return
        for metric in METRICS:
            row = _best(selected, metric)
            lines.append(
                f"- {METRIC_LABELS[metric]}：{row['run_id']}，{float(row[metric]):.6f}"
            )
        lines.append("")

    add_winners("总体最佳检测器", test_rows)
    conventional = [
        row for row in test_rows
        if "detector_baseline" in run_by_id[row["run_id"]].groups
        or "baseline" in run_by_id[row["run_id"]].groups
    ]
    cas_rows = [row for row in test_rows if "cas_detr" in run_by_id[row["run_id"]].groups]
    add_winners("基线最佳", conventional)
    add_winners("CaS-DETR 最佳", cas_rows)
    for group, title in (
        ("component", "组件消融最佳版本"),
        ("dynamic_base_ratio", "dynamic base ratio 最佳版本"),
        ("fixed_keep_ratio", "fixed keep ratio 最佳版本"),
        ("moe_capacity", "MoE 容量扫描最佳版本"),
        ("expert_count", "专家数扫描最佳版本"),
        ("root_checkpoint", "独立根目录 checkpoint 最佳版本"),
    ):
        selected = [
            row for row in test_rows if group in run_by_id[row["run_id"]].groups
        ]
        add_winners(title, selected)
    return "\n".join(lines).rstrip() + "\n"


def _atomic_copy(source: Path, destination: Path, token: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{token}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def publish(work_dir: Path, runs: Sequence[RunSpec]) -> None:
    validate_results(work_dir, runs)
    token = work_dir.name
    for protocol in ("dairv2x_vehicle8", "uadetrac_vehicle4"):
        stage = _stage_dir(work_dir, protocol)
        target = EXPERIMENTS_DIR / "reports" / protocol
        for name in ("eval_metrics.csv", "benchmark.csv", "fine_grained_eval_metrics.csv"):
            _atomic_copy(stage / name, target / name, token)
        report = conclusion_markdown(
            protocol,
            _read_csv(stage / "eval_metrics.csv"),
            runs,
        )
        report_stage = stage / "hierarchical_eval_conclusions.md"
        report_stage.write_text(report, encoding="utf-8")
        _atomic_copy(report_stage, target / report_stage.name, token)
    (work_dir / "PUBLISHED").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")


def _smoke_runs(runs: Sequence[RunSpec]) -> List[RunSpec]:
    selected = []
    for framework in ("yolo", "yolox", "fasterrcnn", "deim", "dfine", "casdeim"):
        selected.append(next(run for run in runs if run.framework == framework))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "smoke", "accuracy", "tensorrt", "validate", "publish", "all"),
        default="preflight",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected accuracy work even when complete rows already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, runs = resolve_manifest()
    counts = _expected_by_protocol(runs)
    print(f"[PREFLIGHT] resolved {len(runs)} unique runs: {counts}", flush=True)
    if args.phase == "preflight":
        for run in runs:
            print(
                f"{run.protocol}\t{run.framework}\t{run.run_id}\t"
                f"{run.checkpoint_sha256[:12]}\t{Path(run.checkpoint).name}"
            )
        return

    if args.work_dir:
        work_dir = args.work_dir.resolve()
        if not work_dir.is_dir():
            raise FileNotFoundError(work_dir)
    else:
        timestamp = _timestamp()
        backup = _backup_reports(timestamp)
        work_dir = _new_work_dir(timestamp, manifest, runs)
        print(f"[BACKUP] {backup}", flush=True)
    _write_current_manifest_snapshot(work_dir, manifest, runs)
    prune_superseded_rows(work_dir, runs)
    print(f"[WORKDIR] {work_dir}", flush=True)

    selected = list(runs)
    if args.phase == "smoke":
        selected = _smoke_runs(runs)
    if args.run_id:
        wanted = set(args.run_id)
        selected = [run for run in selected if run.run_id in wanted]
        missing = wanted - {run.run_id for run in selected}
        if missing:
            raise KeyError(f"unknown selected run ids: {sorted(missing)}")
    if args.limit is not None:
        selected = selected[: args.limit]

    if args.phase in ("smoke", "accuracy", "all"):
        for index, run in enumerate(selected, 1):
            print(f"[ACCURACY {index}/{len(selected)}] {run.run_id}", flush=True)
            run_accuracy(work_dir, run, force=args.force)
    if args.phase in ("tensorrt", "all"):
        for index, run in enumerate(selected, 1):
            print(f"[TENSORRT {index}/{len(selected)}] {run.run_id}", flush=True)
            run_benchmark(work_dir, run)
    if args.phase in ("validate", "publish", "all"):
        validate_results(work_dir, runs)
        print(
            f"[VALIDATE] all {len(runs)} runs and {2 * len(runs)} val/test rows passed",
            flush=True,
        )
    if args.phase in ("publish", "all"):
        publish(work_dir, runs)
        print("[PUBLISH] Vehicle8/Vehicle4 reports replaced atomically", flush=True)


if __name__ == "__main__":
    main()

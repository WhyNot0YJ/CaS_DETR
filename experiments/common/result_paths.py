"""集中管理跨实验结果 CSV 的路径和追加写入。"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent


def reports_dir() -> Path:
    override = os.environ.get("EXPERIMENT_RESULTS_DIR")
    if override:
        return Path(override).expanduser()
    protocol = os.environ.get(
        "EXPERIMENT_DATASET_PROTOCOL", "dairv2x_vehicle5"
    ).lower()
    return EXPERIMENTS_DIR / "reports" / protocol


def result_csv(kind: str) -> Path:
    """Return the single shared CSV for one result category."""
    if kind not in {
        "results",
        "eval_metrics",
        "fine_grained_eval_metrics",
        "benchmark",
    }:
        raise ValueError(f"unsupported result category: {kind}")
    directory = reports_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{kind}.csv"


def append_csv_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Append rows and widen the header when a later producer adds columns."""
    rows = [dict(row) for row in rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    existing_fields = []
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fields = list(existing_fields)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        return

    # Rewriting is necessary only when a new column appears; otherwise append.
    if existing_rows and fields != existing_fields:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not existing_rows and (not path.exists() or path.stat().st_size == 0):
            writer.writeheader()
        writer.writerows(rows)


def upsert_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    key_fields: Sequence[str],
) -> None:
    """Replace rows with the same key instead of duplicating benchmark retries."""
    rows = [dict(row) for row in rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    fields = []
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    incoming_keys = {
        tuple(str(row.get(field, "")) for field in key_fields)
        for row in rows
    }
    existing_rows = [
        row
        for row in existing_rows
        if tuple(str(row.get(field, "")) for field in key_fields) not in incoming_keys
    ]
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)


def update_csv_rows(
    path: Path,
    *,
    match: Mapping[str, object],
    updates: Mapping[str, object],
) -> int:
    """Update matching rows and widen the CSV for newly added result columns."""
    if not path.exists() or not path.stat().st_size:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    count = 0
    for row in rows:
        if all(str(row.get(field, "")) == str(value) for field, value in match.items()):
            row.update(updates)
            count += 1
    if not count:
        return 0
    for field in updates:
        if field not in fields:
            fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return count


def run_metadata(*, run_id: str, framework: str, model: str, dataset: str, seed: object = "") -> Dict[str, object]:
    return {
        "run_id": run_id,
        "framework": framework,
        "experiment": run_id,
        "model": model,
        "dataset": dataset,
        "seed": seed,
    }

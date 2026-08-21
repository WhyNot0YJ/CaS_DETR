#!/usr/bin/env python3
"""Split legacy mixed report CSVs into dataset-protocol subdirectories."""

from __future__ import annotations

import csv
from pathlib import Path


REPORTS = Path(__file__).resolve().parent.parent / "reports"
CSV_NAMES = ("results.csv", "eval_metrics.csv", "benchmark.csv")


def row_protocol(row: dict[str, str]) -> str:
    dataset = str(row.get("dataset", "")).lower()
    if "uadetrac" in dataset or "ua-detrac" in dataset or dataset == "data":
        return "uadetrac"
    return "dairv2x"


def migrate_csv(path: Path) -> None:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row_protocol(row), []).append(row)
    for protocol, protocol_rows in grouped.items():
        target = REPORTS / protocol / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(protocol_rows)
    path.unlink()


def main() -> None:
    for name in CSV_NAMES:
        migrate_csv(REPORTS / name)
    legacy_plot = REPORTS / "results.png"
    if legacy_plot.is_file():
        target = REPORTS / "dairv2x" / legacy_plot.name
        target.parent.mkdir(parents=True, exist_ok=True)
        legacy_plot.replace(target)


if __name__ == "__main__":
    main()

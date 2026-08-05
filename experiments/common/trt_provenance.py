"""TensorRT artifact provenance and conservative engine reuse checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from common.hierarchical_eval import sha256_file


def engine_provenance_path(engine: str | Path) -> Path:
    engine_path = Path(engine)
    return engine_path.with_suffix(engine_path.suffix + ".provenance.json")


def build_engine_provenance(
    *,
    checkpoint: str | Path,
    config: str | Path,
    framework: str,
    export_options: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    config_path = Path(config).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "framework": str(framework),
        "precision": "fp16",
        "export_options": dict(export_options),
    }


def artifact_hash_suffix(provenance: Mapping[str, Any], length: int = 12) -> str:
    checkpoint_hash = str(provenance["checkpoint_sha256"])
    config_hash = str(provenance["config_sha256"])
    return f"ckpt-{checkpoint_hash[:length]}_cfg-{config_hash[:length]}"


def engine_is_reusable(engine: str | Path, expected: Mapping[str, Any]) -> bool:
    engine_path = Path(engine)
    sidecar = engine_provenance_path(engine_path)
    if not engine_path.is_file() or not sidecar.is_file():
        return False
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == dict(expected)


def write_engine_provenance(engine: str | Path, provenance: Mapping[str, Any]) -> Path:
    sidecar = engine_provenance_path(engine)
    sidecar.write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sidecar

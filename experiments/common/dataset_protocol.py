"""Dataset taxonomy protocols shared by all experiment stacks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_PROTOCOLS = {
    "dairv2x": "dairv2x_vehicle5",
    "uadetrac": "uadetrac_vehicle1",
}
PROTOCOLS = (
    "dairv2x_vehicle5",
    "dairv2x_vehicle8",
    "uadetrac_vehicle1",
    "uadetrac_vehicle4",
)
PROTOCOL_SPECS = {
    "dairv2x_vehicle5": {
        "dataset": "dairv2x",
        "root": Path("/root/autodl-fs/datasets/DAIR-V2X-Vehicle5"),
        "num_classes": 5,
        "suffix": "vehicle5",
    },
    "dairv2x_vehicle8": {
        "dataset": "dairv2x",
        "root": Path("/root/autodl-fs/datasets/DAIR-V2X"),
        "num_classes": 8,
        "suffix": "vehicle8",
    },
    "uadetrac_vehicle1": {
        "dataset": "uadetrac",
        "root": Path("/root/autodl-fs/datasets/UA-DETRAC-Vehicle1"),
        "num_classes": 1,
        "suffix": "vehicle1",
    },
    "uadetrac_vehicle4": {
        "dataset": "uadetrac",
        "root": Path("/root/autodl-fs/datasets/UA-DETRAC_COCO"),
        "num_classes": 4,
        "suffix": "vehicle4",
    },
}


def normalize_protocol(value: str | None, dataset: str) -> str:
    protocol = str(value or DEFAULT_PROTOCOLS[dataset]).lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported dataset protocol: {protocol!r}")
    expected_dataset = PROTOCOL_SPECS[protocol]["dataset"]
    if expected_dataset != dataset:
        raise ValueError(
            f"protocol {protocol!r} cannot be used with dataset {dataset!r}"
        )
    return protocol


def set_report_protocol(protocol: str) -> str:
    protocol = str(protocol).lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported report protocol: {protocol!r}")
    os.environ["EXPERIMENT_DATASET_PROTOCOL"] = protocol
    return protocol


def _load_with_includes(path: Path, loaded: set[Path] | None = None) -> Dict[str, Any]:
    path = path.resolve()
    loaded = loaded or set()
    if path in loaded:
        return {}
    loaded.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: Dict[str, Any] = {}
    for include in data.get("__include__", []):
        include_path = Path(include).expanduser()
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        _merge(merged, _load_with_includes(include_path, loaded))
    _merge(merged, data)
    return merged


def _merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def _detect_dataset(config_path: Path, resolved: Dict[str, Any]) -> str | None:
    text = f"{config_path} {resolved}".lower()
    if "uadetrac" in text or "ua-detrac" in text:
        return "uadetrac"
    if "dairv2x" in text or "dair-v2x" in text or "dair_v2x" in text:
        return "dairv2x"
    return None


def _protocol_output_dir(value: str, protocol: str) -> str:
    value = str(value).replace("\\", "/")
    for suffix in ("_vehicle1", "_vehicle4", "_vehicle5", "_vehicle8"):
        if value.rsplit("/", 1)[-1].endswith(suffix):
            value = value[: -len(suffix)]
            break

    # Keep experiment groups below a protocol namespace, matching YOLO's
    # ``logs/<protocol>/<run>`` layout.  This also prevents a protocol suffix
    # from leaking into the experiment name when an old config is reused.
    parts = value.split("/")
    for index, part in enumerate(parts):
        if part not in {"outputs", "output"}:
            continue
        tail = parts[index + 1 :]
        if tail and tail[0] in PROTOCOLS:
            tail = tail[1:]
        return "/".join(parts[: index + 1] + [protocol] + tail)

    # Configs should use an outputs/ root; retain a deterministic namespace
    # for custom paths rather than reverting to the legacy suffix convention.
    return "/".join([protocol, value.lstrip("/")])


def protocol_output_path(value: str | Path, protocol: str) -> str:
    """Place an explicitly supplied output path below its protocol namespace."""
    protocol = str(protocol).lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported dataset protocol: {protocol!r}")
    return _protocol_output_dir(str(value), protocol)


def protocol_output_dir(config_path: str | Path, protocol: str | None) -> str:
    config_path = Path(config_path)
    resolved = _load_with_includes(config_path)
    value = resolved.get("output_dir", "")
    if not value:
        return ""
    dataset = _detect_dataset(config_path, resolved)
    if dataset is None:
        return str(value)
    resolved_protocol = normalize_protocol(protocol, dataset)
    return _protocol_output_dir(str(value), resolved_protocol)


def _dataset_update(
    root: Path,
    dataset: str,
    split: str,
    *,
    rtdetr_layout: bool,
) -> Dict[str, Any]:
    if rtdetr_layout:
        return {"data_root": str(root), "img_folder": str(root)}
    img_folder = root if dataset == "dairv2x" else root / split
    return {
        "img_folder": str(img_folder),
        "ann_file": str(root / "annotations" / f"instances_{split}.json"),
    }


def apply_detr_protocol_overrides(
    update_dict: Dict[str, Any],
    config_path: str | Path,
    protocol: str | None,
    *,
    rtdetr_layout: bool = False,
) -> str:
    """Apply the detected dataset's taxonomy, paths and output namespace."""
    config_path = Path(config_path)
    resolved = _load_with_includes(config_path)
    dataset = _detect_dataset(config_path, resolved)
    if dataset is None:
        if protocol is None:
            return ""
        return set_report_protocol(str(protocol))

    protocol = set_report_protocol(normalize_protocol(protocol, dataset))
    spec = PROTOCOL_SPECS[protocol]
    root = Path(spec["root"])
    _merge(
        update_dict,
        {
            "num_classes": spec["num_classes"],
            "train_dataloader": {
                "dataset": _dataset_update(
                    root, dataset, "train", rtdetr_layout=rtdetr_layout
                )
            },
            "val_dataloader": {
                "dataset": _dataset_update(
                    root, dataset, "val", rtdetr_layout=rtdetr_layout
                )
            },
        },
    )
    if protocol == "dairv2x_vehicle5" and "cross_domain_eval" in resolved:
        _merge(update_dict, {"cross_domain_eval": {"enable": False}})
    if "output_dir" not in update_dict and resolved.get("output_dir"):
        update_dict["output_dir"] = _protocol_output_dir(
            str(resolved["output_dir"]), protocol
        )
    return protocol

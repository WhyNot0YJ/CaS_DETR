#!/usr/bin/env python3
"""
Post-training evaluation for DEIM / D-FINE models.
Produces CaS_DETR-compatible eval_metrics.csv with the same metric columns.

Usage (from experiments/ directory):
  python3 common/eval_deim_dfine.py \\
      --framework deim \\
      --config DEIM/configs/deim_dfine/deim_hgnetv2_s_dairv2x_no_decoder_ffn_pretrain.yml \\
      --resume DEIM/outputs/deim_hgnetv2_s_dairv2x/best_stg2.pth \\
      --model-name deim_hgnetv2_s \\
      --dataset-name DAIR-V2X

After loading weights, runs ``run_detr_benchmark`` for GFLOPs, Params, FPS, and latency,
then val and optional test metrics and ``eval_metrics.csv``.
"""

import os
import sys
import json
import argparse
import gc
import logging
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.det_eval_metrics import (
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    canonical_category_metric_name,
    compute_weather_subset_metrics,
    extract_per_category_ap_from_coco_eval,
    run_coco_bbox_eval,
    write_eval_csv,
)
from common.detr_eval_utils import log_detr_eval_summary, run_detr_benchmark
from common.result_paths import result_csv, run_metadata
from common.dataset_protocol import apply_detr_protocol_overrides
from common.train_notifications import notify_training_entry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)


class RouterStatsCollector:
    """Accumulate per-layer MoE routing and keep-ratio statistics."""

    def __init__(self):
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.keep_ratio_sum = 0.0
        self.keep_ratio_count = 0

    def update(self, model, outputs):
        encoder_info = outputs.get("encoder_info", {}) if isinstance(outputs, dict) else {}
        ratio = encoder_info.get("dynamic_keep_ratio")
        if isinstance(ratio, torch.Tensor):
            self.keep_ratio_sum += float(ratio.detach().float().sum().item())
            self.keep_ratio_count += int(ratio.numel())

        for name, module in _unwrap_module(model).named_modules():
            logits_cache = getattr(module, "router_logits_cache", None)
            indices_cache = getattr(module, "expert_indices_cache", None)
            if not logits_cache or not indices_cache:
                continue
            try:
                num_experts = int(module.num_experts)
                state = self.layers.setdefault(name, {
                    "num_experts": num_experts,
                    "top_k": int(module.top_k),
                    "token_count": 0,
                    "prob_sum": [0.0] * num_experts,
                    "dispatch_count": [0] * num_experts,
                    "entropy_sum": 0.0,
                })
                for logits, indices in zip(logits_cache, indices_cache):
                    probs = torch.softmax(logits.detach().float(), dim=-1)
                    state["token_count"] += int(probs.shape[0])
                    prob_sum = probs.sum(dim=0).cpu().tolist()
                    state["prob_sum"] = [a + float(b) for a, b in zip(state["prob_sum"], prob_sum)]
                    counts = torch.bincount(indices.detach().reshape(-1), minlength=num_experts).cpu().tolist()
                    state["dispatch_count"] = [a + int(b) for a, b in zip(state["dispatch_count"], counts)]
                    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    state["entropy_sum"] += float(entropy.sum().item())
            finally:
                # Router diagnostics are per-forward data. Do not retain the
                # previous batch's GPU tensors while evaluating the next one.
                reset = getattr(module, "reset_cache", None)
                if callable(reset):
                    reset()

    def to_dict(self):
        layers = []
        for name, state in sorted(self.layers.items()):
            tokens = max(1, int(state["token_count"]))
            dispatches = max(1, sum(state["dispatch_count"]))
            mean_prob = [value / tokens for value in state["prob_sum"]]
            usage = [value / dispatches for value in state["dispatch_count"]]
            balance = state["num_experts"] * sum(a * b for a, b in zip(usage, mean_prob))
            layers.append({
                "layer": name,
                "num_experts": state["num_experts"],
                "top_k": state["top_k"],
                "token_count": state["token_count"],
                "expert_usage": usage,
                "router_probability": mean_prob,
                "router_entropy": state["entropy_sum"] / tokens,
                "expert_load": state["dispatch_count"],
                "load_balance_loss": balance,
            })
        return {
            "average_keep_ratio": (
                self.keep_ratio_sum / self.keep_ratio_count if self.keep_ratio_count else None
            ),
            "layers": layers,
        }

def _unwrap_module(m: Any) -> Any:
    return m.module if hasattr(m, "module") else m


def _set_caip_static_keep_eval(model: Any, enabled: bool) -> Dict[str, Any]:
    """Toggle HybridEncoder.caip_static_keep_eval: fixed token_keep_ratio under eval, CAIP still ranks."""
    base = _unwrap_module(model)
    enc = getattr(base, "encoder", None)
    if enc is None or not hasattr(enc, "caip_static_keep_eval"):
        return {"found": False}
    prev = bool(getattr(enc, "caip_static_keep_eval"))
    setattr(enc, "caip_static_keep_eval", bool(enabled))
    return {"found": True, "prev": prev}


def _set_caip_eval_keep_ratio(model: Any, keep_ratio: float) -> Dict[str, Any]:
    """Override the pruner base ratio for a fixed-keep evaluation only."""
    base = _unwrap_module(model)
    enc = getattr(base, "encoder", None)
    pruner = getattr(enc, "shared_token_pruner", None) if enc is not None else None
    if pruner is None or not hasattr(pruner, "keep_ratio"):
        return {"found": False}
    prev = float(getattr(pruner, "keep_ratio"))
    setattr(pruner, "keep_ratio", float(keep_ratio))
    return {"found": True, "prev": prev}


def _set_prune_in_eval(model: Any, enabled: bool) -> Dict[str, Any]:
    """Enable/disable token pruning during eval by toggling TokenLevelPruner.prune_in_eval.

    Returns a dict for restoring original values.
    """
    base = _unwrap_module(model)
    enc = getattr(base, "encoder", None)
    pruner = getattr(enc, "shared_token_pruner", None) if enc is not None else None
    if pruner is None or not hasattr(pruner, "prune_in_eval"):
        return {"found": False}
    prev = bool(getattr(pruner, "prune_in_eval"))
    setattr(pruner, "prune_in_eval", bool(enabled))
    return {"found": True, "prev": prev}


def _set_encoder_epoch(model: Any, epoch: int) -> bool:
    """Sync epoch-dependent encoder behavior (e.g. CAIP warmup scheduling)."""
    base = _unwrap_module(model)
    enc = getattr(base, "encoder", None)
    if enc is None or not hasattr(enc, "set_epoch"):
        return False
    try:
        enc.set_epoch(int(epoch))
        return True
    except Exception:
        return False

def _resolve_resume_path(resume: Optional[str]) -> Optional[str]:
    """Resolve checkpoint path before ``os.chdir`` into DEIM or D-FINE.

    Relative paths such as ``DEIM/outputs/...`` are interpreted from ``experiments/``,
    not from the framework subdirectory after ``chdir``.
    """
    if not resume:
        return resume
    p = Path(resume)
    if p.is_absolute():
        return str(p)
    cand = (EXPERIMENTS_DIR / resume).resolve()
    if cand.is_file():
        return str(cand)
    cand = (Path.cwd() / resume).resolve()
    if cand.is_file():
        return str(cand)
    return str((EXPERIMENTS_DIR / resume).resolve())


# ---------------------------------------------------------------------------
# Framework helpers
# ---------------------------------------------------------------------------

def _setup_deim(
    config_path: str,
    resume: str,
    framework_dir: str = "DEIM",
    overrides: Optional[Dict[str, Any]] = None,
):
    fw_dir = EXPERIMENTS_DIR / framework_dir
    sys.path.insert(0, str(fw_dir))
    saved_cwd = os.getcwd()
    os.chdir(fw_dir)

    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(config_path, resume=resume, **(overrides or {}))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    return solver, cfg, saved_cwd


def _setup_dfine(
    config_path: str,
    resume: str,
    overrides: Optional[Dict[str, Any]] = None,
):
    fw_dir = EXPERIMENTS_DIR / "D-FINE"
    sys.path.insert(0, str(fw_dir))
    saved_cwd = os.getcwd()
    os.chdir(fw_dir)

    from src.core import YAMLConfig
    from src.solver import TASKS

    cfg = YAMLConfig(config_path, resume=resume, **(overrides or {}))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    return solver, cfg, saved_cwd


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _dataset_label2category_map(data_loader):
    """Walk wrappers (e.g. Subset) to find CocoDetection.label2category."""
    ds = data_loader.dataset
    for _ in range(8):
        m = getattr(ds, "label2category", None)
        if m is not None:
            return m
        ds = getattr(ds, "dataset", None)
        if ds is None:
            break
    return None


def _resolve_test_ann_file(ann_file: str) -> Optional[str]:
    """Infer test annotation path from the configured val annotation path."""
    if not ann_file:
        return None
    ann_path = Path(ann_file)
    candidates = []
    name = ann_path.name
    if "instances_val" in name:
        candidates.append(ann_path.with_name(name.replace("instances_val", "instances_test")))
    if "instances_train" in name:
        candidates.append(ann_path.with_name(name.replace("instances_train", "instances_test")))
    if name != "instances_test.json":
        candidates.append(ann_path.with_name("instances_test.json"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_test_img_folder(img_folder: str) -> str:
    """Infer test image folder while keeping dataset roots that already contain all images."""
    if not img_folder:
        return img_folder
    img_path = Path(img_folder)
    if img_path.name in {"val", "train"}:
        test_dir = img_path.with_name("test")
        if test_dir.exists():
            return str(test_dir)
    return img_folder


def _build_test_dataloader(cfg) -> Tuple[Optional[Any], Optional[str]]:
    """Build an eval-only test dataloader by cloning val_dataloader config."""
    val_cfg = cfg.yaml_cfg.get("val_dataloader", {})
    dataset_cfg = val_cfg.get("dataset", {})
    test_ann = _resolve_test_ann_file(dataset_cfg.get("ann_file", ""))
    if not test_ann:
        return None, None

    test_loader_cfg = deepcopy(val_cfg)
    test_loader_cfg["dataset"] = deepcopy(dataset_cfg)
    test_loader_cfg["dataset"]["ann_file"] = test_ann
    test_loader_cfg["dataset"]["img_folder"] = _resolve_test_img_folder(
        test_loader_cfg["dataset"].get("img_folder", "")
    )

    cfg.yaml_cfg["test_dataloader"] = test_loader_cfg
    loader = cfg.build_dataloader("test_dataloader")
    return loader, test_ann


def _dataset_image_root_hint(data_loader) -> str:
    ds = getattr(data_loader, "dataset", None)
    for _ in range(8):
        if ds is None:
            break
        for attr in ("img_folder", "root"):
            v = getattr(ds, attr, None)
            if isinstance(v, str) and v:
                return v
        ds = getattr(ds, "dataset", None)
    return ""


def _safe_len(obj) -> int:
    try:
        return int(len(obj))
    except Exception:
        return -1


def _log_split_context(split_name: str, data_loader: Any, ann_file: str) -> None:
    ds_len = _safe_len(getattr(data_loader, "dataset", None))
    dl_len = _safe_len(data_loader)
    img_root = _dataset_image_root_hint(data_loader)
    l2c = _dataset_label2category_map(data_loader)
    LOG.info(
        "[split=%s] ann_file=%s | ann_exists=%s | dataset_len=%s | dataloader_len=%s | img_root=%s | has_label2category=%s",
        split_name,
        ann_file,
        bool(ann_file and Path(ann_file).exists()),
        ds_len,
        dl_len,
        img_root or "(unknown)",
        l2c is not None,
    )


@torch.no_grad()
def collect_predictions(
    model,
    postprocessor,
    data_loader,
    device,
    *,
    remap_mscoco_category: bool = False,
    label2category=None,
    router_stats=None,
) -> List[Dict]:
    """Run inference and return predictions in COCO detection format."""
    model.eval()
    all_preds: List[Dict] = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        outputs = model(samples)
        if router_stats is not None:
            router_stats.update(model, outputs)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_sizes)

        for target, result in zip(targets, results):
            img_id = int(target["image_id"].flatten()[0].item())
            boxes = result["boxes"].cpu()
            scores = result["scores"].cpu()
            labels = result["labels"].cpu()

            # xyxy -> xywh
            xywh = boxes.clone()
            xywh[:, 2] -= xywh[:, 0]
            xywh[:, 3] -= xywh[:, 1]

            for j in range(len(scores)):
                lid = int(labels[j].item())
                # PostProcessor with remap_mscoco_category=True already emits COCO category ids.
                # Otherwise labels are train indices 0..N-1 — map via dataset (same as CocoEvaluatorTrainLabelMapping / CaS +1 for contiguous 1..N).
                if remap_mscoco_category:
                    cat_id = lid
                elif label2category is not None:
                    cat_id = int(label2category[lid])
                else:
                    cat_id = lid

                all_preds.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": xywh[j].tolist(),
                    "score": scores[j].item(),
                })
    return all_preds


# ---------------------------------------------------------------------------
# CaS-compatible metric computation
# ---------------------------------------------------------------------------

def _build_coco_gt_dict(ann_file: str) -> Dict[str, Any]:
    """Read COCO annotation JSON and return as dict (for pycocotools)."""
    with open(ann_file, "r", encoding="utf-8") as f:
        gt = json.load(f)
    gt.setdefault("info", {"description": "eval", "version": "1.0", "year": 2025})
    return gt


def _is_dair_dataset(dataset_name: str) -> bool:
    low = dataset_name.lower()
    return "dair" in low or "dairv2x" in low


def compute_cas_metrics(
    ann_file: str,
    predictions: List[Dict],
    dataset_name: str,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Full CaS_DETR-compatible metrics from GT + predictions.

    Returns ``(metrics, class_names, weather_buckets)``. ``weather_buckets`` is the
    sorted list of weather subset names found in ``images[].weather`` (empty if none).
    Per-bucket entries land in ``metrics`` as ``weather_<bucket>_mAP_50/5095``.
    """
    coco_gt = _build_coco_gt_dict(ann_file)
    categories = sorted(coco_gt.get("categories", []), key=lambda c: c["id"])
    class_names = [str(c["name"]) for c in categories]

    ce = run_coco_bbox_eval(coco_gt, predictions)
    if ce is None:
        empty = {k: 0.0 for k in [
            "mAP_0.5", "mAP_0.75", "mAP_0.5_0.95",
            "AP_small", "AP_medium", "AP_large",
            "AP_small_50", "AP_medium_50", "AP_large_50",
        ]}
        return empty, class_names, []

    stats = ce.stats
    s50, m50, l50 = coco_area_ap_at_iou50(ce)

    metrics: Dict[str, Any] = {
        "mAP_0.5": float(stats[1]),
        "mAP_0.75": float(stats[2]),
        "mAP_0.5_0.95": float(stats[0]),
        "AP_small": float(stats[3]) if len(stats) > 3 else 0.0,
        "AP_medium": float(stats[4]) if len(stats) > 4 else 0.0,
        "AP_large": float(stats[5]) if len(stats) > 5 else 0.0,
        "AP_small_50": s50,
        "AP_medium_50": m50,
        "AP_large_50": l50,
    }

    per50, per5095 = extract_per_category_ap_from_coco_eval(ce, categories)
    for name, v in per50.items():
        metrics[f"AP50_{canonical_category_metric_name(name)}"] = v
    for name, v in per5095.items():
        metrics[f"AP5095_{canonical_category_metric_name(name)}"] = v

    weather_metrics = compute_weather_subset_metrics(coco_gt, predictions)
    metrics.update(weather_metrics)
    weather_buckets = sorted({
        k[len("weather_"):-len("_mAP_50")]
        for k in weather_metrics
        if k.endswith("_mAP_50")
    })

    return metrics, class_names, weather_buckets


def _config_stub_for_benchmark(yaml_cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Minimal config dict for ``run_detr_benchmark`` / ``model_display_name``."""
    ds = yaml_cfg.get("train_dataloader", {}).get("dataset", {})
    return {
        "data": {
            "data_root": str(ds.get("data_root", "")),
            "dataset_class": str(ds.get("type", "")),
        },
        "model": {},
        "_config_path": config_path,
    }


def _resolve_cd_config_path(cd_yaml: str) -> Optional[Path]:
    """Resolve cross_domain_eval.config which is normally relative to a framework dir
    (e.g. DQM-DETR/), but eval_deim_dfine.py chdirs into that dir before calling here,
    so the relative path is already valid from cwd. Falls back to absolute and to
    the framework dir explicitly if needed."""
    p = Path(cd_yaml)
    if p.is_absolute() and p.is_file():
        return p
    cand = Path.cwd() / p
    if cand.is_file():
        return cand
    return None


def _apply_cross_domain_remap_to_dict(coco_gt_dict: Dict[str, Any], label_map: Dict[int, int],
                                      drop_unmapped: bool, override_categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mirror of ``CocoEvaluatorTrainLabelMapping._apply_cross_domain_remap`` but on a plain dict.

    Rewrites GT category_id from val-domain to train-domain ids and replaces categories
    with ``override_categories`` (so prediction label index 0..N-1 maps to the train head)."""
    out = deepcopy(coco_gt_dict)
    kept = []
    for ann in out.get("annotations", []):
        src = int(ann["category_id"])
        if src in label_map:
            ann["category_id"] = label_map[src]
            kept.append(ann)
        elif not drop_unmapped:
            kept.append(ann)
    out["annotations"] = kept
    out["categories"] = [dict(c) for c in override_categories]
    return out


def _maybe_run_cross_domain_csv_row(
    *,
    cfg,
    model,
    postprocessor,
    device,
    csv_path: Path,
    model_name: str,
    bench_dict: Optional[Dict[str, Any]],
) -> bool:
    """If ``cross_domain_eval`` is configured, run inference on its val set and write a DAWN CSV row.

    Returns True iff a row was written.
    """
    yaml_cfg = getattr(cfg, "yaml_cfg", {}) or {}
    cd_cfg = yaml_cfg.get("cross_domain_eval")
    if not cd_cfg or not cd_cfg.get("enable", True):
        return False

    cd_yaml_rel = cd_cfg.get("config")
    if not cd_yaml_rel:
        LOG.info("[CrossDomain/CSV] cross_domain_eval.config not set, skipping")
        return False
    cd_yaml = _resolve_cd_config_path(cd_yaml_rel)
    if cd_yaml is None:
        LOG.warning("[CrossDomain/CSV] cross-domain config not found: %s", cd_yaml_rel)
        return False

    # cfg's YAMLConfig was already imported into the framework subpackage (DQM-DETR / DEIM / etc.)
    # via _setup_deim/_setup_dfine. Reuse the same module to keep registry parity.
    from engine.core import YAMLConfig as _YAMLConfig  # type: ignore[import-not-found]
    cd = _YAMLConfig(str(cd_yaml))

    cd_loader = cd.val_dataloader
    cd_ds = cd.yaml_cfg.get("val_dataloader", {}).get("dataset", {}) or {}
    ann_file = cd_ds.get("ann_file", "")
    if not ann_file or not Path(ann_file).is_file():
        LOG.warning("[CrossDomain/CSV] DAWN ann_file missing: %s", ann_file)
        return False

    eval_cfg = cd.yaml_cfg.get("evaluator", {}) or {}
    label_map_raw = eval_cfg.get("cross_domain_label_map") or {}
    label_map = {int(k): int(v) for k, v in label_map_raw.items()}
    drop_unmapped = bool(eval_cfg.get("drop_unmapped_gt", False))
    override_categories = eval_cfg.get("override_categories") or []

    LOG.info("[CrossDomain/CSV] running DAWN cross-domain eval | ann=%s | label_map=%s",
             ann_file, label_map)
    preds = collect_predictions(
        model,
        postprocessor,
        cd_loader,
        device,
        remap_mscoco_category=False,
        label2category=None,
    )
    LOG.info("[CrossDomain/CSV] collected %d predictions", len(preds))

    coco_gt = _build_coco_gt_dict(ann_file)
    if label_map and override_categories:
        coco_gt = _apply_cross_domain_remap_to_dict(coco_gt, label_map, drop_unmapped, override_categories)

    # Write a temp ann file for compute_cas_metrics (it re-reads JSON). Keep the function
    # API stable by serializing the remapped GT to a sibling temp path.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix="_cd_remapped.json", delete=False, encoding="utf-8") as fh:
        json.dump(coco_gt, fh)
        tmp_ann = fh.name
    try:
        metrics, class_names, weather_buckets = compute_cas_metrics(tmp_ann, preds, "DAWN")
    finally:
        try:
            os.remove(tmp_ann)
        except OSError:
            pass

    log_detr_eval_summary(LOG, "cross_domain_dawn", metrics, bench_dict)
    dawn_csv_path = csv_path
    write_eval_csv(
        dawn_csv_path,
        model=model_name,
        dataset="DAWN",
        eval_split="cross_domain_dawn",
        metrics=metrics,
        class_names=class_names,
        append=True,
        benchmark=bench_dict,
        weather_buckets=weather_buckets or None,
    )
    LOG.info("[CrossDomain/CSV] Wrote %s", dawn_csv_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_best_checkpoint(output_dir: str) -> Optional[str]:
    """Search for best checkpoint in common save locations."""
    d = Path(output_dir)
    for name in ("best_stg2.pth", "best.pth", "best_stg1.pth", "last.pth"):
        p = d / name
        if p.exists():
            return str(p)
    return None


@notify_training_entry(
    "正式评测",
    enabled_env="TRAIN_NOTIFY_FINAL_EVAL",
    framework_from_cli=True,
)
def main():
    parser = argparse.ArgumentParser(description="CaS-compatible eval for DEIM / CaS-DETR / DQM-DETR / D-FINE")
    parser.add_argument("--framework", required=True, choices=["deim", "casdeim", "dqmdeim", "dfine"],
                        help="Which framework (deim, casdeim, dqmdeim or dfine)")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path. If omitted, auto-detect from output_dir in config.")
    parser.add_argument("--model-name", default=None,
                        help="Model display name for CSV (default: config file stem)")
    parser.add_argument("--run-id", default=None,
                        help="Stable CSV run identifier (default: derived from output_dir)")
    parser.add_argument("--seed", default=None,
                        help="Seed recorded in the evaluation CSV metadata")
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset display name for CSV (auto-detect from config paths)")
    parser.add_argument("--output-csv", default=None,
                        help="CSV path (default: experiments/reports/<protocol>/eval_metrics.csv)")
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--dairv2x", dest="dataset_protocol", action="store_const", const="dairv2x")
    protocol.add_argument("--uadetrac", dest="dataset_protocol", action="store_const", const="uadetrac")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-pytorch-benchmark",
        action="store_true",
        help="Skip diagnostic PyTorch speed timing; TensorRT results are recorded separately.",
    )
    parser.add_argument("--splits", default="val,test",
                        help="Comma-separated eval splits to run (default: val,test)")
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
        help="Override evaluation DataLoader workers (does not change predictions).",
    )
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable token pruning during evaluation (sets TokenLevelPruner.prune_in_eval=False).",
    )
    parser.add_argument(
        "--caip-static-keep-eval",
        action="store_true",
        help="When CAIP is on, use fixed keep ratio token_keep_ratio in eval mode; CAIP scores still rank tokens.",
    )
    parser.add_argument(
        "--caip-eval-keep-ratio",
        type=float,
        default=None,
        help="Fixed CAIP keep ratio for eval only; requires --caip-static-keep-eval.",
    )
    parser.add_argument(
        "--epoch",
        default=None,
        type=int,
        help="Encoder epoch for epoch-dependent behaviors (default: checkpoint last_epoch).",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory for predictions_<split>.json (default: output_dir).",
    )
    parser.add_argument(
        "--router-stats",
        default=None,
        help="Optional JSON path for per-layer MoE router statistics.",
    )
    args = parser.parse_args()
    if args.caip_eval_keep_ratio is not None:
        if not args.caip_static_keep_eval:
            parser.error("--caip-eval-keep-ratio requires --caip-static-keep-eval")
        if not 0.0 < args.caip_eval_keep_ratio <= 1.0:
            parser.error("--caip-eval-keep-ratio must be in (0, 1]")

    config_path = str(Path(args.config).resolve())
    model_name = args.model_name or Path(args.config).stem
    protocol_overrides: Dict[str, Any] = {}
    resolved_protocol = apply_detr_protocol_overrides(
        protocol_overrides, config_path, args.dataset_protocol
    )
    if args.eval_num_workers is not None:
        if args.eval_num_workers < 0:
            parser.error("--eval-num-workers must be non-negative")
        protocol_overrides.setdefault("val_dataloader", {})["num_workers"] = (
            args.eval_num_workers
        )

    # Auto-detect dataset name from config path
    dataset_name = args.dataset_name
    if dataset_name is None:
        low = config_path.lower()
        if resolved_protocol == "dairv2x":
            dataset_name = "DAIR-V2X"
        elif resolved_protocol == "uadetrac":
            dataset_name = "UA-DETRAC"
        elif "dairv2x" in low or "dair-v2x" in low or "dair_v2x" in low:
            dataset_name = "DAIR-V2X"
        elif "uadetrac" in low or "ua-detrac" in low or "ua_detrac" in low:
            dataset_name = "UA-DETRAC"
        else:
            dataset_name = "unknown"

    LOG.info("Framework: %s | Config: %s | Dataset: %s", args.framework, config_path, dataset_name)

    if args.resume:
        args.resume = _resolve_resume_path(args.resume)
        LOG.info("Resolved --resume to %s", args.resume)

    # Setup framework
    if args.framework == "deim":
        solver, cfg, saved_cwd = _setup_deim(
            config_path, args.resume or "", framework_dir="DEIM",
            overrides=protocol_overrides,
        )
    elif args.framework == "casdeim":
        solver, cfg, saved_cwd = _setup_deim(
            config_path, args.resume or "", framework_dir="CaS-DETR",
            overrides=protocol_overrides,
        )
    elif args.framework == "dqmdeim":
        solver, cfg, saved_cwd = _setup_deim(
            config_path, args.resume or "", framework_dir="DQM-DETR",
            overrides=protocol_overrides,
        )
    else:
        solver, cfg, saved_cwd = _setup_dfine(
            config_path, args.resume or "", overrides=protocol_overrides
        )

    # Resolve checkpoint, then solver.eval(): runs _setup() so solver has model/ema and loads weights.
    # Without eval(), DetSolver never runs BaseSolver._setup(), so attributes like solver.ema do not exist.
    resume_path = args.resume
    if not resume_path:
        output_dir = cfg.yaml_cfg.get("output_dir", "./outputs")
        ckpt = _find_best_checkpoint(output_dir)
        if ckpt is None:
            LOG.error("No checkpoint found in %s. Use --resume to specify.", output_dir)
            sys.exit(1)
        LOG.info("Auto-detected checkpoint: %s", ckpt)
        resume_path = ckpt
    if resume_path:
        resume_path = _resolve_resume_path(resume_path) or resume_path
    cfg.resume = resume_path
    solver.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = solver.ema.module if solver.ema else solver.model
    model.to(device)
    model.eval()
    # Ensure CAIP / encoder epoch-dependent scheduling matches training epoch.
    enc_epoch = int(args.epoch) if args.epoch is not None else int(getattr(solver, "last_epoch", 0))
    if _set_encoder_epoch(model, enc_epoch):
        LOG.info("Synced encoder epoch for eval: %s", enc_epoch)
    else:
        LOG.info("Encoder has no set_epoch(); skip epoch sync (requested=%s).", enc_epoch)

    restore_keep_ratio: Dict[str, Any] = {"found": False}
    if args.caip_eval_keep_ratio is not None:
        restore_keep_ratio = _set_caip_eval_keep_ratio(model, args.caip_eval_keep_ratio)
        if not restore_keep_ratio.get("found"):
            LOG.error("Requested fixed CAIP keep ratio, but no TokenLevelPruner was found.")
            sys.exit(2)
        LOG.info(
            "CAIP eval: keep ratio %.3f (previous %.3f)",
            args.caip_eval_keep_ratio,
            restore_keep_ratio.get("prev"),
        )

    restore_static_keep: Dict[str, Any] = {"found": False}
    if args.caip_static_keep_eval:
        restore_static_keep = _set_caip_static_keep_eval(model, enabled=True)
        if restore_static_keep.get("found"):
            LOG.info(
                "CAIP eval: fixed keep ratio token_keep_ratio (caip_static_keep_eval: %s -> True)",
                restore_static_keep.get("prev"),
            )
        else:
            LOG.info("Requested --caip-static-keep-eval, but encoder has no caip_static_keep_eval (skip).")

    restore_pruning: Dict[str, Any] = {"found": False}
    if args.disable_pruning:
        restore_pruning = _set_prune_in_eval(model, enabled=False)
        if restore_pruning.get("found"):
            LOG.info("Disabled token pruning in eval (prune_in_eval: %s -> False)", restore_pruning.get("prev"))
        else:
            LOG.info("Requested --disable-pruning, but no TokenLevelPruner found on model.encoder.shared_token_pruner")

    yaml_cfg = getattr(cfg, "yaml_cfg", {}) or {}
    remap_mscoco = bool(yaml_cfg.get("remap_mscoco_category", False))

    cfg_stub = _config_stub_for_benchmark(yaml_cfg, config_path)
    bench_dict = (
        None
        if args.skip_pytorch_benchmark
        else run_detr_benchmark(model, cfg_stub, args.framework, device, LOG)
    )

    # Write CSV
    output_dir = Path(cfg.yaml_cfg.get("output_dir", "./outputs"))
    csv_path = Path(args.output_csv) if args.output_csv else result_csv("eval_metrics")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    primary_ann = str(
        cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("ann_file", "")
    )
    primary_name = Path(primary_ann).name
    primary_split = (
        "eval" if primary_name == "instances_eval.json"
        else "test" if primary_name == "instances_test.json"
        else "val"
    )
    seen_splits: set[str] = set()
    append_csv = csv_path.exists()
    wrote_any = False
    test_metrics: Dict[str, Any] = {}
    router_stats = RouterStatsCollector() if args.router_stats else None

    for split_name in split_names:
        if split_name == "val" and primary_split != "val":
            split_name = primary_split
        if split_name in seen_splits:
            continue
        seen_splits.add(split_name)
        if split_name in {"val", "eval"}:
            data_loader = solver.val_dataloader
            split_cfg = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
            ann_file = split_cfg.get("ann_file", "")
        elif split_name == "test":
            val_cfg = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
            LOG.info(
                "[test] inferring from val: ann_file=%s | img_folder=%s",
                val_cfg.get("ann_file", ""),
                val_cfg.get("img_folder", ""),
            )
            data_loader, ann_file = _build_test_dataloader(cfg)
            if data_loader is None or not ann_file:
                LOG.info("No test split found for config, skipping test evaluation.")
                continue
        else:
            LOG.warning("Unknown split '%s', skipping.", split_name)
            continue

        _log_split_context(split_name, data_loader, ann_file)
        if not ann_file or not Path(ann_file).exists():
            LOG.warning("Cannot find %s annotation file: %s, skipping.", split_name, ann_file)
            continue

        l2c = _dataset_label2category_map(data_loader)
        LOG.info("Running inference on %s set ...", split_name)
        preds = collect_predictions(
            model,
            solver.postprocessor,
            data_loader,
            device,
            remap_mscoco_category=remap_mscoco,
            label2category=l2c,
            router_stats=router_stats,
        )
        LOG.info("Collected %d predictions for %s", len(preds), split_name)
        predictions_dir = Path(args.predictions_dir) if args.predictions_dir else output_dir
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / f"predictions_{split_name}.json"
        predictions_path.write_text(json.dumps(preds), encoding="utf-8")
        LOG.info("Wrote %s", predictions_path)
        LOG.info("Computing CaS-compatible metrics from %s ...", ann_file)
        metadata = run_metadata(
            run_id=args.run_id or f"{output_dir.parent.name}/{output_dir.name}",
            framework=args.framework,
            model=model_name,
            dataset=dataset_name,
            seed=args.seed if args.seed is not None else cfg.yaml_cfg.get("seed", ""),
        )
        metrics, class_names, weather_buckets = compute_cas_metrics(
            ann_file, preds, dataset_name
        )
        log_detr_eval_summary(LOG, split_name, metrics, bench_dict)

        write_eval_csv(
            csv_path,
            model=model_name,
            dataset=dataset_name,
            eval_split=split_name,
            metrics=metrics,
            class_names=class_names,
            append=append_csv,
            benchmark=bench_dict,
            weather_buckets=weather_buckets or None,
            metadata=metadata,
        )
        append_csv = True
        wrote_any = True
        if split_name == "test":
            test_metrics = metrics

        # Top-300 DETR prediction lists are large. Release each split before
        # constructing the next DataLoader.
        del preds
        gc.collect()

    if wrote_any:
        LOG.info("Wrote %s", csv_path)
    else:
        LOG.warning("No eval split was written to CSV.")

    if router_stats is not None:
        router_path = Path(args.router_stats)
        router_path.parent.mkdir(parents=True, exist_ok=True)
        router_path.write_text(json.dumps(router_stats.to_dict(), indent=2), encoding="utf-8")
        LOG.info("Wrote %s", router_path)

    cd_wrote = _maybe_run_cross_domain_csv_row(
        cfg=cfg,
        model=model,
        postprocessor=solver.postprocessor,
        device=device,
        csv_path=csv_path,
        model_name=model_name,
        bench_dict=bench_dict,
    )
    if cd_wrote:
        wrote_any = True

    if restore_pruning.get("found"):
        _set_prune_in_eval(model, enabled=bool(restore_pruning.get("prev")))
    if restore_static_keep.get("found"):
        _set_caip_static_keep_eval(model, enabled=bool(restore_static_keep.get("prev")))
    if restore_keep_ratio.get("found"):
        _set_caip_eval_keep_ratio(model, float(restore_keep_ratio.get("prev")))

    os.chdir(saved_cwd)
    if os.environ.get("TRAIN_NOTIFY_FINAL_EVAL", "").strip().lower() in {"1", "true", "yes", "on"} and not test_metrics:
        raise RuntimeError("训练后 test 评测未产生指标，未发送成功通知")
    return {
        "output_dir": str(output_dir.resolve()),
        "metrics": test_metrics,
        "metric_source": f"{csv_path} (test)",
    }


if __name__ == "__main__":
    main()

"""Protocol-level small-object low-sample exclusion diagnostics (DAIR / UA).

Covers, for each official protocol:
- excluded categories and diagnostic metric names (spec);
- diagnostic values computed from a synthetic COCOeval precision tensor;
- eval_metrics.csv columns (new aggregated small diagnostics present,
  raw ``AP_small_50`` / ``AP_small_5095`` aggregates suppressed);
- training-completion email metric selection (new columns win over raw ones).
"""

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.det_eval_metrics import (  # noqa: E402
    small_object_diagnostic_area_ap,
    write_eval_csv,
)
from common.small_object_diagnostics import small_object_diagnostic_spec  # noqa: E402
from common.train_notifications import METRIC_LABELS, _metrics_from_row  # noqa: E402

DAIR_NAMES = [
    "car", "truck", "van", "bus",
    "pedestrian", "cyclist", "motorcyclist", "trafficcone",
]
UA_NAMES = ["car", "van", "bus", "others"]


def _synthetic_coco_eval(per_cat_ap50, cat_ids):
    """Build a fake COCOeval: small-bucket AP@0.5 per category, others IoUs 0.

    ``per_cat_ap50`` maps category index (0-based) to its small AP@0.5 value;
    categories absent from the map keep ``-1`` (no small GT), mimicking how
    COCOeval naturally ignores UA ``others`` in the small bucket. The medium
    bucket is filled with 0.99 to catch area-index mistakes.
    """
    n_cat = len(cat_ids)
    precision = np.full((10, 2, n_cat, 4, 3), -1.0)
    precision[:, :, :, 2, 2] = 0.99  # medium bucket sentinel
    for cat_index, ap50 in per_cat_ap50.items():
        precision[0, :, cat_index, 1, 2] = ap50
        precision[1:, :, cat_index, 1, 2] = 0.0
    return SimpleNamespace(
        eval={"precision": precision},
        params=SimpleNamespace(maxDets=[1, 10, 100], catIds=cat_ids),
    )


def _categories(names):
    return [{"id": i + 1, "name": name} for i, name in enumerate(names)]


def _check_csv_columns(dataset_name, class_names, diag_keys, diag_values):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval_metrics.csv"
        metrics = {
            "mAP_50": 0.5,
            "mAP_5095": 0.3,
            "AP_small_50": 0.11,
            "AP_small_5095": 0.05,
            diag_keys[0]: diag_values[0],
            diag_keys[1]: diag_values[1],
        }
        write_eval_csv(
            path,
            "model",
            dataset_name,
            "eval",
            metrics,
            class_names=class_names,
            diagnostic_metric_keys=list(diag_keys),
        )
        row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert diag_keys[0] in row and diag_keys[1] in row
        assert row[diag_keys[0]] == f"{diag_values[0]:.6f}"
        assert row[diag_keys[1]] == f"{diag_values[1]:.6f}"
        # raw aggregated small columns must not be written
        assert "AP_small_50" not in row
        assert "AP_small_5095" not in row
        # per-class small columns stay
        for name in class_names:
            assert f"AP_small_50_{name}" in row
            assert f"AP_small_5095_{name}" in row


def _check_email_prefers_excluded(diag_keys, raw50, raw5095, excl50, excl5095):
    metrics = _metrics_from_row({
        "AP_small_50": raw50,
        "AP_small_5095": raw5095,
        diag_keys[0]: excl50,
        diag_keys[1]: excl5095,
    })
    assert metrics["mapsmall50"] == excl50
    assert metrics["mapsmall5095"] == excl5095


def test_dairv2x_excludes_bus_truck():
    excluded, keys = small_object_diagnostic_spec("DAIR-V2X")
    assert excluded == ("bus", "truck")
    assert keys == ("AP_small_50_excl_bus_truck", "AP_small_5095_excl_bus_truck")

    ap50_by_cat = {0: 0.8, 1: 0.9, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4, 6: 0.3, 7: 0.2}
    categories = _categories(DAIR_NAMES)
    coco_eval = _synthetic_coco_eval(ap50_by_cat, [c["id"] for c in categories])

    out = small_object_diagnostic_area_ap(coco_eval, categories, "DAIR-V2X", area_index=1)
    kept = [ap50_by_cat[i] for i in (0, 2, 4, 5, 6, 7)]  # drop truck(1) & bus(3)
    assert np.isclose(out[keys[0]], np.mean(kept))
    assert np.isclose(out[keys[1]], np.mean(kept) / 10.0)

    _check_csv_columns("DAIR-V2X", DAIR_NAMES, keys, (out[keys[0]], out[keys[1]]))
    _check_email_prefers_excluded(keys, 0.10, 0.20, out[keys[0]], out[keys[1]])


def test_uadetrac_excludes_bus_and_ignores_others():
    excluded, keys = small_object_diagnostic_spec("UA-DETRAC")
    assert excluded == ("bus",)
    assert keys == ("AP_small_50_excl_bus", "AP_small_5095_excl_bus")

    # others(3) has no small GT -> stays -1 and is naturally ignored by COCOeval
    ap50_by_cat = {0: 0.8, 1: 0.5, 2: 0.2}
    categories = _categories(UA_NAMES)
    coco_eval = _synthetic_coco_eval(ap50_by_cat, [c["id"] for c in categories])

    out = small_object_diagnostic_area_ap(coco_eval, categories, "UA-DETRAC", area_index=1)
    kept = [ap50_by_cat[0], ap50_by_cat[1]]  # drop bus(2); others auto-ignored
    assert np.isclose(out[keys[0]], np.mean(kept))
    assert np.isclose(out[keys[1]], np.mean(kept) / 10.0)

    _check_csv_columns("UA-DETRAC", UA_NAMES, keys, (out[keys[0]], out[keys[1]]))
    _check_email_prefers_excluded(keys, 0.10, 0.20, out[keys[0]], out[keys[1]])


def test_unknown_protocol_has_no_diagnostic():
    assert small_object_diagnostic_spec("COCO") == ((), ())
    assert small_object_diagnostic_area_ap(None, [], "COCO") == {}


def test_email_labels_mention_low_sample_exclusion():
    labels = dict(METRIC_LABELS)
    assert "exclude <20 small GT)" in labels["mapsmall50"]
    assert "exclude <20 small GT)" in labels["mapsmall5095"]


def test_yolo_and_deim_share_the_common_spec():
    """Both final-eval paths must consume the shared rule module (no copies)."""
    root = Path(__file__).resolve().parents[1]
    yolo_src = (root / "yolo" / "base_yolo_trainer.py").read_text(encoding="utf-8")
    assert "from common.small_object_diagnostics import small_object_diagnostic_spec" in yolo_src
    assert "small_object_diagnostic_area_ap" in yolo_src
    assert "diagnostic_metric_keys" in yolo_src

    deim_src = (root / "common" / "eval_deim_dfine.py").read_text(encoding="utf-8")
    assert "from common.small_object_diagnostics import small_object_diagnostic_spec" in deim_src
    assert "small_object_diagnostic_area_ap" in deim_src
    assert "DAIR_SMALL_EXCLUDED_CATEGORIES = " not in deim_src


def _rtdetr_style_batch(class_names, excluded_names, drop_names):
    """Synthetic RT-DETR-chain eval: small GT for all classes except ``drop_names``
    (which have no GT at all); perfect predictions except for ``excluded_names``
    (GT present, no predictions -> per-class small AP 0)."""
    categories = _categories(class_names)
    targets, preds = [], []
    for i, cat in enumerate(categories):
        if cat["name"] in drop_names:
            continue
        box = [10.0 + i * 40, 10.0, 20.0, 20.0]  # area 400 < 32^2 -> small
        targets.append({
            "image_id": 0, "category_id": cat["id"], "bbox": box, "area": 400.0,
            "iscrowd": 0,
        })
        if cat["name"] in excluded_names:
            continue
        preds.append({
            "image_id": 0, "category_id": cat["id"], "bbox": box, "score": 0.99,
        })
    return categories, preds, targets


def test_rtdetr_chain_applies_exclusion():
    """Official RT-DETR v2 chain: compute_cas_style_map_metrics must emit the
    same diagnostic columns via the shared rule (fairness across frameworks)."""
    try:
        from common.cas_style_map_metrics import compute_cas_style_map_metrics
    except Exception as exc:  # pycocotools unavailable
        print(f"skip rtdetr-chain functional test: {exc}")
        return

    _, dair_keys = small_object_diagnostic_spec("DAIR-V2X")
    cats, preds, tgts = _rtdetr_style_batch(DAIR_NAMES, {"bus", "truck"}, set())
    metrics = compute_cas_style_map_metrics(
        preds, tgts, cats, img_h=720, img_w=1280,
        print_per_category=True, dataset_name="DAIR-V2X",
    )
    assert np.isclose(metrics[dair_keys[0]], 1.0), metrics.get(dair_keys[0])
    assert np.isclose(metrics[dair_keys[1]], 1.0), metrics.get(dair_keys[1])
    # excluded classes keep their (bad) per-class small AP; only the diagnostic
    # aggregate excludes them
    assert metrics["AP_small_50_bus"] == 0.0
    assert metrics["AP_small_5095_truck"] == 0.0

    _, ua_keys = small_object_diagnostic_spec("UA-DETRAC")
    cats, preds, tgts = _rtdetr_style_batch(UA_NAMES, {"bus"}, {"others"})
    metrics = compute_cas_style_map_metrics(
        preds, tgts, cats, img_h=540, img_w=960,
        print_per_category=True, dataset_name="UA-DETRAC",
    )
    assert np.isclose(metrics[ua_keys[0]], 1.0), metrics.get(ua_keys[0])
    assert np.isclose(metrics[ua_keys[1]], 1.0), metrics.get(ua_keys[1])

    # RT-DETR CSV writer forwards the diagnostic columns
    root = Path(__file__).resolve().parents[1]
    cas_eval_src = (root / "common" / "rtdetr_cas_eval.py").read_text(encoding="utf-8")
    assert "dataset_name=ds_name" in cas_eval_src
    assert "diagnostic_metric_keys=small_diag_keys" in cas_eval_src
    utils_src = (root / "common" / "detr_eval_utils.py").read_text(encoding="utf-8")
    assert "diagnostic_metric_keys=diagnostic_metric_keys" in utils_src


if __name__ == "__main__":
    test_dairv2x_excludes_bus_truck()
    test_uadetrac_excludes_bus_and_ignores_others()
    test_unknown_protocol_has_no_diagnostic()
    test_email_labels_mention_low_sample_exclusion()
    test_yolo_and_deim_share_the_common_spec()
    test_rtdetr_chain_applies_exclusion()
    print("small-object diagnostics self-check passed")

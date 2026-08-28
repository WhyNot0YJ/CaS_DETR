"""Small self-checks for the shared per-class COCO report fields."""

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.det_eval_metrics import (  # noqa: E402
    coco_area_ap_excluding_categories,
    extract_per_category_ap_from_coco_eval,
    write_eval_csv,
)
from common.small_object_diagnostics import small_object_diagnostic_spec  # noqa: E402


def test_per_class_small_ap_and_csv_columns():
    precision = np.full((10, 2, 1, 4, 3), -1.0)
    precision[:, :, 0, 0, 2] = 0.8
    precision[:, :, 0, 1, 2] = 0.3
    precision[0, :, 0, 1, 2] = 0.2
    coco_eval = SimpleNamespace(
        eval={"precision": precision},
        params=SimpleNamespace(maxDets=[1, 10, 100], catIds=[1]),
    )
    categories = [{"id": 1, "name": "Car"}]

    ap50, ap5095 = extract_per_category_ap_from_coco_eval(coco_eval, categories, area_index=1)
    assert np.isclose(ap50["car"], 0.2)
    assert np.isclose(ap5095["car"], 0.29)

    excluded_precision = np.full((10, 2, 3, 4, 3), -1.0)
    excluded_precision[:, :, :, 1, 2] = np.array([0.2, 0.8, 0.6])[None, None, :]
    excluded_eval = SimpleNamespace(
        eval={"precision": excluded_precision},
        params=SimpleNamespace(maxDets=[1, 10, 100], catIds=[1, 2, 3]),
    )
    excluded_categories = [
        {"id": 1, "name": "Car"},
        {"id": 2, "name": "Bus"},
        {"id": 3, "name": "Truck"},
    ]
    excl50, excl5095 = coco_area_ap_excluding_categories(
        excluded_eval, excluded_categories, ("bus", "truck")
    )
    assert np.isclose(excl50, 0.2)
    assert np.isclose(excl5095, 0.2)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_eval_csv(
            path, "model", "dataset", "eval",
            {
                "AP_small_50_car": ap50["car"],
                "AP_small_5095_car": ap5095["car"],
                "AP_small_50_excl_bus_truck": 0.42,
            },
            class_names=["Car"],
            diagnostic_metric_keys=["AP_small_50_excl_bus_truck"],
        )
        row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert row["AP_small_50_car"] == "0.200000"
        assert row["AP_small_5095_car"] == "0.290000"
        assert row["AP_small_50_excl_bus_truck"] == "0.420000"
        assert "AP_small_50" not in row
        assert "AP_small_5095" not in row

        legacy_path = Path(tmp) / "legacy_metrics.csv"
        legacy_path.write_text(
            "model,dataset,eval_split,AP_small_50,AP_small_5095\n"
            "old_model,dataset,eval,0.100000,0.200000\n",
            encoding="utf-8",
        )
        write_eval_csv(
            legacy_path,
            "new_model",
            "dataset",
            "eval",
            {"mAP_50": 0.50},
        )
        legacy_rows = list(csv.DictReader(legacy_path.open(newline="", encoding="utf-8")))
        assert "AP_small_50" not in legacy_rows[0]
        assert "AP_small_5095" not in legacy_rows[0]


def test_small_object_diagnostic_protocol_rules():
    assert small_object_diagnostic_spec("DAIR-V2X") == (
        ("bus", "truck"),
        ("AP_small_50_excl_bus_truck", "AP_small_5095_excl_bus_truck"),
    )
    assert small_object_diagnostic_spec("UA-DETRAC") == (
        ("bus",),
        ("AP_small_50_excl_bus", "AP_small_5095_excl_bus"),
    )
    assert small_object_diagnostic_spec("unknown") == ((), ())


if __name__ == "__main__":
    test_per_class_small_ap_and_csv_columns()
    test_small_object_diagnostic_protocol_rules()
    print("det_eval_metrics small-AP self-check passed")

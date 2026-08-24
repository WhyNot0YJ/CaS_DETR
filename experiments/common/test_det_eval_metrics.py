"""Small self-checks for the shared per-class COCO report fields."""

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.det_eval_metrics import (  # noqa: E402
    extract_per_category_ap_from_coco_eval,
    write_eval_csv,
)


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

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_eval_csv(
            path, "model", "dataset", "eval",
            {"AP_small_50_car": ap50["car"], "AP_small_5095_car": ap5095["car"]},
            class_names=["Car"],
        )
        row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert row["AP_small_50_car"] == "0.200000"
        assert row["AP_small_5095_car"] == "0.290000"


if __name__ == "__main__":
    test_per_class_small_ap_and_csv_columns()
    print("det_eval_metrics small-AP self-check passed")

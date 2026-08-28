"""Self-checks for the metrics shown in training-completion emails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.train_notifications import _metrics_from_row  # noqa: E402


def test_small_metric_email_prefers_low_sample_exclusion():
    metrics = _metrics_from_row({
        "AP_small_50": 0.10,
        "AP_small_5095": 0.20,
        "AP_small_50_excl_bus": 0.30,
        "AP_small_5095_excl_bus": 0.40,
    })
    assert metrics["mapsmall50"] == 0.30
    assert metrics["mapsmall5095"] == 0.40


if __name__ == "__main__":
    test_small_metric_email_prefers_low_sample_exclusion()
    print("train notification metrics self-check passed")

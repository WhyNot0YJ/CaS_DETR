"""Regression checks for DEIM/D-FINE final-evaluation split resolution."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval_deim_dfine import _resolve_test_ann_file  # noqa: E402


def test_configured_test_annotation_is_reused():
    with tempfile.TemporaryDirectory() as tmp:
        ann_file = Path(tmp) / "instances_test.json"
        ann_file.write_text("{}", encoding="utf-8")
        assert _resolve_test_ann_file(str(ann_file)) == str(ann_file)


if __name__ == "__main__":
    test_configured_test_annotation_is_reused()
    print("eval_deim_dfine split-resolution self-check passed")

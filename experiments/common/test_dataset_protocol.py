from __future__ import annotations

import os
import unittest
from pathlib import Path

from dataset_protocol import (
    apply_detr_protocol_overrides,
    protocol_output_dir,
    protocol_output_path,
    set_report_protocol,
)


EXPERIMENTS = Path(__file__).resolve().parent.parent
CAS_DAIR = (
    EXPERIMENTS
    / "CaS-DETR/configs/dataset/ablation/archive/"
    "cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml"
)
CAS_UA = (
    EXPERIMENTS
    / "CaS-DETR/configs/dataset/ablation/archive/"
    "cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_uadetrac.yml"
)


class DatasetProtocolTest(unittest.TestCase):
    def test_dairv2x_is_the_default(self):
        update = {}
        protocol = apply_detr_protocol_overrides(update, CAS_DAIR, None)

        self.assertEqual(protocol, "dairv2x")
        self.assertEqual(update["num_classes"], 8)
        self.assertIn("/DAIR-V2X/annotations/", update["train_dataloader"]["dataset"]["ann_file"])
        self.assertTrue(update["val_dataloader"]["dataset"]["ann_file"].endswith("instances_eval.json"))
        self.assertEqual(
            update["output_dir"],
            "./outputs/dairv2x/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x",
        )

    def test_uadetrac_is_the_default(self):
        previous = os.environ.get("EXPERIMENT_DATASET_PROTOCOL")
        try:
            update = {}
            protocol = apply_detr_protocol_overrides(update, CAS_UA, None)

            self.assertEqual(protocol, "uadetrac")
            self.assertEqual(update["num_classes"], 4)
            self.assertIn(
                "UA-DETRAC_COCO",
                update["val_dataloader"]["dataset"]["ann_file"],
            )
            self.assertTrue(
                update["val_dataloader"]["dataset"]["ann_file"].endswith("instances_test.json")
            )
            self.assertIn("./outputs/uadetrac/", protocol_output_dir(CAS_UA, None))
        finally:
            if previous is None:
                os.environ.pop("EXPERIMENT_DATASET_PROTOCOL", None)
            else:
                os.environ["EXPERIMENT_DATASET_PROTOCOL"] = previous

    def test_rtdetr_uses_the_protocol_eval_split(self):
        update = {}
        apply_detr_protocol_overrides(update, CAS_UA, None, rtdetr_layout=True)
        dataset = update["val_dataloader"]["dataset"]
        self.assertEqual(dataset["split"], "test")
        self.assertTrue(dataset["img_folder"].endswith("UA-DETRAC_COCO/test"))

        update = {}
        apply_detr_protocol_overrides(update, CAS_DAIR, None, rtdetr_layout=True)
        self.assertEqual(update["val_dataloader"]["dataset"]["split"], "eval")

    def test_report_protocol_accepts_all_report_namespaces(self):
        previous = os.environ.get("EXPERIMENT_DATASET_PROTOCOL")
        try:
            self.assertEqual(
                set_report_protocol("uadetrac"), "uadetrac"
            )
            self.assertEqual(
                os.environ["EXPERIMENT_DATASET_PROTOCOL"], "uadetrac"
            )
        finally:
            if previous is None:
                os.environ.pop("EXPERIMENT_DATASET_PROTOCOL", None)
            else:
                os.environ["EXPERIMENT_DATASET_PROTOCOL"] = previous

    def test_explicit_output_path_is_namespaced(self):
        self.assertEqual(
            protocol_output_path("outputs/batch_r18_dairv2x", "dairv2x"),
            "outputs/dairv2x/batch_r18_dairv2x",
        )


if __name__ == "__main__":
    unittest.main()

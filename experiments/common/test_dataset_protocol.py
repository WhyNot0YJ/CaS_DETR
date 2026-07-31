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
    / "CaS-DETR/configs/dataset/ablation/"
    "cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml"
)
CAS_UA = (
    EXPERIMENTS
    / "CaS-DETR/configs/dataset/ablation/"
    "cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_uadetrac.yml"
)


class DatasetProtocolTest(unittest.TestCase):
    def test_vehicle5_is_the_dair_default(self):
        update = {}
        protocol = apply_detr_protocol_overrides(update, CAS_DAIR, None)

        self.assertEqual(protocol, "dairv2x_vehicle5")
        self.assertEqual(update["num_classes"], 5)
        self.assertIn("DAIR-V2X-Vehicle5", update["train_dataloader"]["dataset"]["ann_file"])
        self.assertEqual(
            update["output_dir"],
            "./outputs/dairv2x_vehicle5/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x",
        )

    def test_vehicle8_is_an_explicit_legacy_override(self):
        update = {}
        protocol = apply_detr_protocol_overrides(
            update, CAS_DAIR, "dairv2x_vehicle8"
        )

        self.assertEqual(protocol, "dairv2x_vehicle8")
        self.assertEqual(update["num_classes"], 8)
        self.assertIn("/DAIR-V2X/annotations/", update["val_dataloader"]["dataset"]["ann_file"])
        self.assertEqual(
            update["output_dir"],
            "./outputs/dairv2x_vehicle8/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x",
        )

    def test_uadetrac_vehicle1_is_the_default(self):
        previous = os.environ.get("EXPERIMENT_DATASET_PROTOCOL")
        try:
            update = {}
            protocol = apply_detr_protocol_overrides(update, CAS_UA, None)

            self.assertEqual(protocol, "uadetrac_vehicle1")
            self.assertEqual(update["num_classes"], 1)
            self.assertIn(
                "UA-DETRAC-Vehicle1",
                update["val_dataloader"]["dataset"]["ann_file"],
            )
            self.assertIn("./outputs/uadetrac_vehicle1/", protocol_output_dir(CAS_UA, None))
        finally:
            if previous is None:
                os.environ.pop("EXPERIMENT_DATASET_PROTOCOL", None)
            else:
                os.environ["EXPERIMENT_DATASET_PROTOCOL"] = previous

    def test_report_protocol_accepts_all_report_namespaces(self):
        previous = os.environ.get("EXPERIMENT_DATASET_PROTOCOL")
        try:
            self.assertEqual(
                set_report_protocol("uadetrac_vehicle4"), "uadetrac_vehicle4"
            )
            self.assertEqual(
                os.environ["EXPERIMENT_DATASET_PROTOCOL"], "uadetrac_vehicle4"
            )
        finally:
            if previous is None:
                os.environ.pop("EXPERIMENT_DATASET_PROTOCOL", None)
            else:
                os.environ["EXPERIMENT_DATASET_PROTOCOL"] = previous

    def test_explicit_output_path_is_namespaced(self):
        self.assertEqual(
            protocol_output_path("outputs/batch_r18_dairv2x", "dairv2x_vehicle5"),
            "outputs/dairv2x_vehicle5/batch_r18_dairv2x",
        )


if __name__ == "__main__":
    unittest.main()

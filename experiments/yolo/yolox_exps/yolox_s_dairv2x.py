#!/usr/bin/env python3
"""YOLOX-S on the native DAIR-V2X protocol."""

import os

from cas_yolox_exp import CasYoloxExp


class Exp(CasYoloxExp):
    def __init__(self):
        super().__init__()
        self.image_layout = "flat_image_dir"
        self.val_ann = "instances_eval.json"
        self.test_ann = "instances_eval.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

#!/usr/bin/env python3
"""YOLOX-M on the native UA-DETRAC protocol."""

import os

from cas_yolox_exp import CasYoloxExp


class Exp(CasYoloxExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.67
        self.width = 0.75
        self.num_classes = 1
        self.val_ann = "instances_test.json"
        self.test_ann = "instances_test.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

#!/usr/bin/env python3
"""YOLOX-S on the native UA-DETRAC protocol."""

import os

from cas_yolox_exp import CasYoloxExp


class Exp(CasYoloxExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

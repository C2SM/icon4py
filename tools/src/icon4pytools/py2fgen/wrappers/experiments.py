# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from enum import Enum


class UninitialisedArrays(Enum):
    # these arrays are not initialised in this experiment and not used
    # therefore unpacking needs to be skipped as otherwise it will trigger
    # an error.
    APE_R02B04 = ("ape_r02b04", ["mask_hdiff", "zd_diffcoef", "zd_vertoffset", "zd_intcoef", "hdef_ic", "div_ic", "dwdx", "dwdy"])

    def __init__(self, experiment: str, arrays: list[str]):
        self._experiment = experiment
        self._arrays = arrays

    @property
    def experiment(self) -> str:
        return self._experiment

    @property
    def arrays(self) -> list[str]:
        return self._arrays

    @classmethod
    def get_arrays_for_experiment(cls, experiment_name: str) -> list[str]:
        for ua in cls:
            if ua.experiment == experiment_name:
                return ua.arrays
        return []

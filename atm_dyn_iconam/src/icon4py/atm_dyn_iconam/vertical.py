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

import numpy as np


class VerticalGridConfig:
    num_k_levels = 65


class VerticalModelConfig:
    rayleigh_damping_height = 12500  # height of rayleigh damping in [m]
    vct_a = []  # read from vertical_coord_tables


class VerticalModelParams:
    def __init__(self, vertical_model_config: VerticalModelConfig):
        self.index_of_damping_height = np.argmax(
            vertical_model_config.vct_a >= vertical_model_config.rayleigh_damping_height
        )

    def get_index_of_damping_layer(self):
        return self.index_of_damping_height

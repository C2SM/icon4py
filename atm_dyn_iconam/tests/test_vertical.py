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

import math

import numpy as np
import pytest

from icon4py.atm_dyn_iconam.vertical import (
    VerticalModelConfig,
    VerticalModelParams,
)


@pytest.mark.parametrize(
    "max_h,damping,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109000, 45000, 123)],
)
def test_nrdmax_calculation(max_h, damping, delta):
    vertical_model_config = VerticalModelConfig()
    vertical_model_config.rayleigh_damping_height = damping
    vertical_model_config.vct_a = np.arange(0, max_h, delta)
    vertical_params = VerticalModelParams(vertical_model_config=vertical_model_config)
    assert vertical_params.get_index_of_damping_layer() == math.ceil(damping / delta)

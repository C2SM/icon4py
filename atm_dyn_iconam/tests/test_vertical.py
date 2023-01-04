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
import os

import numpy as np
import pytest

from icon4py.diffusion.icon_grid import VerticalModelParams
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


@pytest.mark.parametrize(
    "max_h,damping,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109050, 45000, 123)],
)
def test_nrdmax_calculation(max_h, damping, delta):
    vct_a = np.arange(0, max_h, delta)
    vct_a = vct_a[::-1]
    vertical_params = VerticalModelParams(rayleigh_damping_height=damping, vct_a=vct_a)
    assert (
        vertical_params.get_index_of_damping_layer()
        == vct_a.shape[0] - math.ceil(damping / delta) - 1
    )


def test_nrdmax_calculation_from_icon_input(icon_grid):
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    sp = IconSerialDataProvider(
        "icon_diffusion_init", data_path, True
    ).from_savepoint_init(linit=True, date="2021-06-20T12:00:10.000")
    a = sp.vct_a()
    damping_height = 12500
    vertical_params = VerticalModelParams(
        rayleigh_damping_height=damping_height, vct_a=a
    )
    assert 9 == vertical_params.get_index_of_damping_layer()
    a_array = np.asarray(a)
    assert a_array[9] > damping_height
    assert a_array[10] < damping_height

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
import pytest

@pytest.mark.datatest
def test_verify_geofac_n2s_field_manipulation(interpolation_savepoint, icon_grid):
    geofac_n2s = np.asarray(interpolation_savepoint.geofac_n2s())
    int_state = interpolation_savepoint.construct_interpolation_state_for_diffusion()
    geofac_c = np.asarray(int_state.geofac_n2s_c)
    geofac_nbh = np.asarray(int_state.geofac_n2s_nbh)
    assert np.count_nonzero(geofac_nbh) > 0
    cec_table = icon_grid.get_c2cec_connectivity().table
    assert np.allclose(geofac_c, geofac_n2s[:, 0])
    assert geofac_nbh[cec_table].shape == geofac_n2s[:, 1:].shape
    assert np.allclose(geofac_nbh[cec_table], geofac_n2s[:, 1:])

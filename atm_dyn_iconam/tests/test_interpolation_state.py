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
def test_cecdim(interpolation_savepoint, icon_grid):
    interpolation_fields = interpolation_savepoint.construct_interpolation_state()
    geofac_n2s = np.asarray(interpolation_fields.geofac_n2s)
    geofac_n2s_nbh = np.asarray(interpolation_fields.geofac_n2s_nbh)
    assert np.count_nonzero(geofac_n2s_nbh) > 0
    c2cec = icon_grid.get_c2cec_connectivity().table
    ported = geofac_n2s_nbh[c2cec]
    assert ported.shape == geofac_n2s[:, 1:].shape
    assert np.allclose(ported, geofac_n2s[:, 1:])

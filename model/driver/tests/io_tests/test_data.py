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
import icon4py.model.common.grid.simple as simple_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils import helpers
from icon4py.model.driver.io import data, ugrid


def test_data_array_has_ugrid_and_cf_attributes():
    grid = simple_grid.SimpleGrid()
    buffer = helpers.random_field(grid, dims.CellDim, dims.KDim)
    data_array = data.to_data_array(buffer, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"])
    assert data_array.attrs["units"] == "kg m-3"
    assert data_array.attrs["standard_name"] == "air_density"
    assert data_array.attrs["icon_var_name"] == "rho"
    assert data_array.attrs["long_name"].startswith("density")
    ## attributes for mapping to ugrid
    assert data_array.attrs["location"] == "face"
    assert data_array.attrs["coordinates"] == "clon clat"
    assert data_array.attrs["mesh"] == ugrid.MESH

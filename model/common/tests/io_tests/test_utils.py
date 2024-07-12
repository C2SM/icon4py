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
import gt4py.next as gtx
import xarray as xa

import icon4py.model.common.grid.simple as simple_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.io import ugrid, utils
from icon4py.model.common.states import data, model
from icon4py.model.common.test_utils import helpers


def test_data_array_has_ugrid_and_cf_attributes():
    grid = simple_grid.SimpleGrid()
    buffer = helpers.random_field(grid, dims.CellDim, dims.KDim)
    data_array = utils.to_data_array(buffer, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"])
    assert data_array.attrs["units"] == "kg m-3"
    assert data_array.attrs["standard_name"] == "air_density"
    assert data_array.attrs["icon_var_name"] == "rho"
    assert data_array.attrs["long_name"].startswith("density")
    ## attributes for mapping to ugrid
    assert data_array.attrs["location"] == "face"
    assert data_array.attrs["coordinates"] == "clon clat"
    assert data_array.attrs["mesh"] == ugrid.MESH


def test_type_check_for_datafields():
    grid = simple_grid.SimpleGrid()
    field = helpers.zero_field(grid, dims.CellDim, dims.KDim, dtype=gtx.int32)
    model_field = model.ModelField(data=field, attrs=data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"])
    assert not isinstance(field, model.ModelField)
    assert isinstance(model_field, model.DataField)
    dataarray = utils.to_data_array(field, attrs=model_field.attrs)
    assert isinstance(dataarray, xa.DataArray)
    assert isinstance(dataarray, model.DataField)
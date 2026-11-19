# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import icon4py.model.common.grid.simple as simple_grid
from icon4py.model.common import dimension as dims, field
from icon4py.model.common.framework import quantities as qty
from icon4py.model.common.io import ugrid, utils
from icon4py.model.common.states import model
from icon4py.model.common.utils import data_allocation as data_alloc


def test_data_array_has_ugrid_and_cf_attributes():
    grid = simple_grid.simple_grid()
    buffer = field.Field(qty.RhoOnCellK, data_alloc.random_field(grid, dims.CellDim, dims.KDim))
    data_array = utils.to_data_array(buffer)
    assert data_array.attrs["units"] == "kg m-3"
    assert data_array.attrs["standard_name"] == "air_density"
    assert data_array.attrs["long_name"] == "density"
    ## attributes for mapping to ugrid
    assert data_array.attrs["location"] == "face"
    assert data_array.attrs["coordinates"] == "clon clat"
    assert data_array.attrs["mesh"] == ugrid.MESH
    assert isinstance(data_array, model.DataField)

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

import xa
from gt4py.next import Field
from xarray import DataArray

from icon4py.model.common.dimension import CellDim, KDim


def to_cf_dataset(
    field: Field[[CellDim, KDim], float],
) -> xa.Dataset:
    """Convert a gt4py field to a xarray dataset"""
    attrs = dict(
        standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"
    )
    # air_potential_temperature
    attrs = dict(
        standard_name="virtual_potential_temperature",
        long_name="virtual potential temperature",
        units="K",
        icon_var_name="theta_v",
    )
    attrs = dict(
        standard_name="dimensionless_exner_function", long_name="", units="1", icon_var_name="exner"
    )
    attrs = dict(
        standard_name="upward_air_velocity",
        long_name="vertical velocity",
        units="m s-1",
        icon_var_name="w",
    )
    attrs = dict(
        standard_name="normal_velocity",
        long_name="velocity normal to edge",
        units="m s-1",
        icon_var_name="theta_v",
    )

    attrs = dict(
        standard_name="eastward_wind",
        long_name="eastward wind component",
        units="m s-1",
        icon_var_name="u",
    )
    attrs = dict(
        standard_name="northward_wind",
        long_name="northward wind component",
        units="m s-1",
        icon_var_name="v",
    )

    # prognostics = dict(air_density = None )
    def to_data_array(
        field: [[CellDim, KDim], float], coords: tuple[DataArray, DataArray], attrs: dict
    ) -> xa.DataArray:
        """Convert a gt4py field to a xarray dataset"""

        return xa.DataArray(data=field, coords=coords, dims=["lat", "lon"])

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
import xarray as xa
from gt4py.next.common import Dimension, DimensionKind

from icon4py.model.driver.io.xgrid import dimension_mapping, ugrid_attributes


DEFAULT_CALENDAR = "proleptic_gregorian"

### CF attributes of the prognostic variables
PROGNOSTIC_CF_ATTRIBUTES = dict(
    air_density = dict(standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"), 
    virtual_potential_temperature = dict(standard_name="virtual_potential_temperature", long_name="virtual potential temperature", units="K", icon_var_name="theta_v"),
    exner_function = dict(standard_name="dimensionless_exner_function", long_name="exner function", icon_var_name="exner"),
    upward_air_velocity = dict(standard_name="upward_air_velocity", long_name="vertical air velocity component", units="m s-1", icon_var_name="w"),
    normal_velocity = dict(standard_name="normal_velocity", long_name="velocity normal to edge", units="m s-1", icon_var_name="vn"),
    tangential_velocity = dict(standard_name="tangential_velocity", long_name="velocity tangential to edge", units="m s-1", icon_var_name="vt"),
    )

### CF attributes of diagnostic variables
DIAGNOSTIC_CF_ATTRIBUTES = dict(eastward_wind = dict(
    standard_name="eastward_wind",
    long_name="eastward wind component",
    units="m s-1",
    icon_var_name="u",
), northward_wind=dict(
    standard_name="northward_wind",
    long_name="northward wind component",
    units="m s-1",
    icon_var_name="v",
),)



    

# TODO (halungge): add typing for the field, build from already attributed field
def to_data_array(
    field, attrs: dict = {}
) -> xa.DataArray:
    """Convert a gt4py field to a xarray dataarray"""
    dims = tuple(dimension_mapping[d] for d in field.domain.dims)
    horizontal_dim = list(filter(lambda d: _is_horizontal(d), field.domain.dims))[0]
    uxgrid_attrs = ugrid_attributes(horizontal_dim)
    attrs.update(uxgrid_attrs)
    return xa.DataArray(data=field.ndarray,  dims=dims, attrs=attrs)


def _is_horizontal(dim:Dimension):
    return dim.kind == DimensionKind.HORIZONTAL



class OutputFile:
    def __init__(self, filename: str, grid_ds: xa.Dataset):
        self.filename = filename
        self.grid = grid_ds    
        
        
        
    
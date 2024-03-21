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
from gt4py._core.definitions import ScalarT
from gt4py.next import Field
from gt4py.next.common import DimsT
from typing_extensions import Protocol
from xarray import DataArray

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.grid_manager import GridFile


### CF attributes of the prognostic variables
prognostics_attrs = dict(air_density = dict(standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"), 
virtual_potential_temperature = dict(standard_name="virtual_potential_temperature", long_name="virtual potential temperature", units="K", icon_var_name="theta_v"),
exner_function = dict(standard_name="dimensionless_exner_function", long_name="exner function", icon_var_name="exner"),
upward_air_velocity = dict(standard_name="upward_air_velocity", long_name="vertical air velocity component", units="m s-1", icon_var_name="w"),
normal_velocity = dict(standard_name="normal_velocity", long_name="velocity normal to edge", units="m s-1", icon_var_name="vn"),
tangential_velocity = dict(standard_name="tangential_velocity", long_name="velocity tangential to edge", units="m s-1", icon_var_name="vt"),
                   )
### CF attributes of diagnostic variables
diagnostic_attrs = dict(eastward_wind = dict(
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


class Dimensions(GridFile.DimensionName):
    TIME = "time"
    HEIGHT = "level"
    PRESSURE_LEVEL = "pressure_level"
    
    
class AnnotatedField:
    def __init__(self, field: Protocol[DimsT, ScalarT], name: str, attrs: dict):
        self.field = field
        self.name = name
        self.attrs = attrs
        self.coords = None

    
    
    def to_data_array(self, grid_ds: xa.Dataset) -> xa.DataArray:
        """Convert a gt4py field to a xarray dataset"""
        # TODO (magdalena) need to know the dimension because that determines the data array
        return xa.DataArray(data=self.field.asnumpy(), dims=["lat", "lon", "height"], attrs=self.attrs)
    
    def set_coords(self, coords: tuple[DataArray, DataArray]):
        self.coords = coords  


# create a a xarray dataarray from a gt4py field
def to_data_array(
    field: Field[[CellDim, KDim], float], coords: tuple[DataArray, DataArray], attrs: dict
) -> xa.DataArray:
    """Convert a gt4py field to a xarray dataset"""
    return xa.DataArray(data=field.asnumpy(), coords=coords, dims=["lat", "lon", "height"], attrs=attrs)

class OutputFile:
    def __init__(self, filename: str, grid_ds: xa.Dataset):
        self.filename = filename
        self.grid = grid_ds    
        
        dims = Dimensions()
        
        
    
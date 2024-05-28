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
from typing import Final

import xarray as xa
from gt4py._core.definitions import ScalarT
from gt4py.next.common import Dimension, DimensionKind, Dims, DimsT, Field

from icon4py.model.driver.io.ugrid import dimension_mapping, ugrid_attributes


### CF attributes of the prognostic variables
PROGNOSTIC_CF_ATTRIBUTES: Final[dict] = dict(
    air_density=dict(
        standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"
    ),
    virtual_potential_temperature=dict(
        standard_name="virtual_potential_temperature",
        long_name="virtual potential temperature",
        units="K",
        icon_var_name="theta_v",
    ),
    exner_function=dict(
        standard_name="dimensionless_exner_function",
        long_name="exner function",
        icon_var_name="exner",
        units="1",
    ),
    upward_air_velocity=dict(
        standard_name="upward_air_velocity",
        long_name="vertical air velocity component",
        units="m s-1",
        icon_var_name="w",
    ),
    normal_velocity=dict(
        standard_name="normal_velocity",
        long_name="velocity normal to edge",
        units="m s-1",
        icon_var_name="vn",
    ),
    tangential_velocity=dict(
        standard_name="tangential_velocity",
        long_name="velocity tangential to edge",
        units="m s-1",
        icon_var_name="vt",
    ),
)

### CF attributes of diagnostic variables
DIAGNOSTIC_CF_ATTRIBUTES: Final[dict] = dict(
    eastward_wind=dict(
        standard_name="eastward_wind",
        long_name="eastward wind component",
        units="m s-1",
        icon_var_name="u",
    ),
    northward_wind=dict(
        standard_name="northward_wind",
        long_name="northward wind component",
        units="m s-1",
        icon_var_name="v",
    ),
)


def to_data_array(
    field: Field[Dims[DimsT], ScalarT], attrs=None, is_on_interface: bool = False
) -> xa.DataArray:
    """Convert a gt4py field to a xarray dataarray.

    Args:
        field: gt4py field,
        attrs: optional dictionary of metadata attributes to be added to the dataarray, empty by default.
        is_on_interface: optional boolean flag indicating if the 2d field is defined on the interface, False by default.
    """
    if attrs is None:
        attrs = {}
    dims = tuple(dimension_mapping(d, is_on_interface) for d in field.domain.dims)
    horizontal_dim = next(filter(lambda d: _is_horizontal(d), field.domain.dims))
    uxgrid_attrs = ugrid_attributes(horizontal_dim)
    attrs.update(uxgrid_attrs)
    return xa.DataArray(data=field.ndarray, dims=dims, attrs=attrs)


def _is_horizontal(dim: Dimension) -> bool:
    return dim.kind == DimensionKind.HORIZONTAL

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final, TypedDict

import gt4py._core.definitions as gt_coredefs
import gt4py.next as gtx
import gt4py.next.common as gt_common
import xarray as xa

from icon4py.model.common.io.ugrid import dimension_mapping, ugrid_attributes


class FieldMetaData(TypedDict):
    standard_name: str
    long_name: str
    units: str
    icon_var_name: str


#: CF attributes of the prognostic variables
PROGNOSTIC_CF_ATTRIBUTES: Final[dict[str, FieldMetaData]] = dict(
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

#: CF attributes of diagnostic variables
DIAGNOSTIC_CF_ATTRIBUTES: Final[dict[str, FieldMetaData]] = dict(
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
    field: gtx.Field[gtx.Dims[gt_common.DimsT], gt_coredefs.ScalarT],
    attrs=None,
    is_on_interface: bool = False,
) -> xa.DataArray:
    """Convert a gt4py field to a xarray data array.

    Args:
        field: gt4py field,
        attrs: optional dictionary of metadata attributes to be added to the data array, empty by default.
        is_on_interface: optional boolean flag indicating if the 2d field is defined on the interface, False by default.
    """
    if attrs is None:
        attrs = {}
    dims = tuple(dimension_mapping(d, is_on_interface) for d in field.domain.dims)
    horizontal_dim = next(filter(lambda d: _is_horizontal(d), field.domain.dims))
    uxgrid_attrs = ugrid_attributes(horizontal_dim)
    attrs.update(uxgrid_attrs)
    return xa.DataArray(data=field.ndarray, dims=dims, attrs=attrs)


def _is_horizontal(dim: gtx.Dimension) -> bool:
    return dim.kind == gtx.DimensionKind.HORIZONTAL

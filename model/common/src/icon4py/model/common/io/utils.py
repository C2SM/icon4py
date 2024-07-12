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
from gt4py import next as gtx
from gt4py._core import definitions as gt_coredefs
from gt4py.next import common as gt_common

from icon4py.model.common.io.ugrid import dimension_mapping, ugrid_attributes


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

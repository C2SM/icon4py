# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import xarray as xa
from gt4py import next as gtx

from icon4py.model.common import field
from icon4py.model.common.io.ugrid import dimension_mapping, ugrid_attributes
from icon4py.model.common.utils import data_allocation as data_alloc


def to_data_array(field_: field.Field, to_host: bool = False) -> xa.DataArray:
    """
    The xarray data array of a typed field, with the CF attributes of its quantity and the UGRID
    attributes of its horizontal dimension.

    `to_host=True` copies the buffer to host (numpy), as netCDF4 cannot consume device arrays.

    Note:
        The data buffer is not copied: the result usually references ``field_`` (a view).
        ``to_host=True`` is a copy only on GPU; on CPU it still references ``field_``.
        Callers that keep the DataArray past the next mutation of ``field_`` must copy it.
    """
    attrs = {
        key: value
        for key in ("standard_name", "long_name", "units")
        if (value := getattr(field_.quantity, key)) is not None
    }
    dims = tuple(dimension_mapping(d) for d in field_.domain.dims)
    horizontal_dim = next(d for d in field_.domain.dims if _is_horizontal(d))
    attrs.update(ugrid_attributes(horizontal_dim))
    data = data_alloc.as_numpy(field_) if to_host else field_.ndarray
    return xa.DataArray(data=data, dims=dims, attrs=attrs)


def _is_horizontal(dim: gtx.Dimension) -> bool:
    return dim.kind == gtx.DimensionKind.HORIZONTAL

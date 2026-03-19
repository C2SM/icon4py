# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import array_api_compat
from gt4py import next as gtx
from gt4py.next import typing as gtx_typing


def flip(field: gtx.Field, dim: gtx.Dimension, allocator: gtx_typing.Allocator) -> gtx.Field:
    """Flip a field along a given dimension.

    Args:
        field: The field to flip.
        dim: The dimension along which to flip the field.
        allocator: Allocator to use for the output field.
    """
    # Note: `allocator` needs to be passed explicitly since GT4Py fields currently don't persist how they were allocated.
    xp = array_api_compat.array_namespace(field.ndarray)
    flipped_array = xp.flip(field.ndarray, axis=field.domain.dims.index(dim))
    return gtx.as_field(field.domain, flipped_array, allocator=allocator)


def index2offset(
    index_field: gtx.Field, dim: gtx.Dimension, allocator: gtx_typing.Allocator
) -> gtx.Field:
    """Convert an index field to an offset field.

    Args:
        index_field: Index field in Python indexing (0-based).
        dim: The dimension along which to convert indices to offsets.
        allocator: Allocator to use for the output field.
    """
    # Note: `allocator` needs to be passed explicitly since GT4Py fields currently don't persist how they were allocated.
    xp = array_api_compat.array_namespace(index_field.ndarray)

    current_index = gtx.as_field(
        gtx.Domain(index_field.domain[dim]),
        xp.arange(
            index_field.domain[dim].unit_range.start,
            index_field.domain[dim].unit_range.stop,
            dtype=index_field.ndarray.dtype,
        ),
        allocator=allocator,
    )
    offset_field = index_field - current_index  # use GT4Py's broadcasting and field arithmetic
    # if GT4Py embedded would propagate the allocator, we could avoid this extra conversion.
    return gtx.as_field(index_field.domain, offset_field.ndarray, allocator=allocator)

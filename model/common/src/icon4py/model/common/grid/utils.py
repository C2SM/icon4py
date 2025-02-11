# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType

import gt4py.next as gtx
import numpy as np
from gt4py.next import Dimension, NeighborTableOffsetProvider


def neighbortable_offset_provider_for_1d_sparse_fields(
    old_shape: tuple[int, int],
    origin_axis: Dimension,
    neighbor_axis: Dimension,
    has_skip_values: bool,
    array_ns: ModuleType = np,
):
    table = array_ns.arange(old_shape[0] * old_shape[1], dtype=gtx.int32).reshape(old_shape)
    assert (
        table.dtype == gtx.int32
    ), 'Neighbor table\'s ("{}" to "{}") data type for 1d sparse fields must be gtx.int32. Instead it\'s "{}"'.format(
        origin_axis, neighbor_axis, table.dtype
    )
    return NeighborTableOffsetProvider(
        table,
        origin_axis,
        neighbor_axis,
        table.shape[1],
        has_skip_values=has_skip_values,
    )

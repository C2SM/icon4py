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

import numpy as np
from gt4py.next import Dimension, NeighborTableOffsetProvider


def neighbortable_offset_provider_for_1d_sparse_fields(
    old_shape: tuple[int, int],
    origin_axis: Dimension,
    neighbor_axis: Dimension,
):
    table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
    return NeighborTableOffsetProvider(table, origin_axis, neighbor_axis, table.shape[1])


class ClassLevelCache:
    _cache = {}

    @staticmethod
    def cache_method(method):
        def wrapper(self, *args, **kwargs):
            key = f"{method.__name__}_{args}_{kwargs}"
            if key not in ClassLevelCache._cache:
                ClassLevelCache._cache[key] = method(self, *args, **kwargs)
            return ClassLevelCache._cache[key]

        return wrapper

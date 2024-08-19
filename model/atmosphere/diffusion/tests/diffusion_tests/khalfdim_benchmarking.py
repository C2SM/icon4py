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

import time

import gt4py.next as gtx
from gt4py.next.ffront.decorator import field_operator

from icon4py.model.common.dimension import KDim, KHalf2K, KHalfDim, Koff
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _copy_khalfdim(f: gtx.Field[[KHalfDim], wpfloat]) -> gtx.Field[[KHalfDim], wpfloat]:
    return f


@field_operator
def _offset_k(f: gtx.Field[[KDim], wpfloat]) -> gtx.Field[[KDim], wpfloat]:
    f_off = f(Koff[-1])
    return f_off


@field_operator
def _convert_khalfdim(f: gtx.Field[[KDim], wpfloat]) -> gtx.Field[[KHalfDim], wpfloat]:
    return f(KHalf2K[1])


def test_benchmarking(grid):
    f = random_field(grid, KHalfDim, dtype=wpfloat)

    f_copy = zero_field(grid, KHalfDim, dtype=wpfloat)
    start_copy = time.time()
    _copy_khalfdim(f, out=f_copy, offset_provider={})
    end_copy = time.time()

    f_k = random_field(grid, KDim, dtype=wpfloat)
    f_offset = zero_field(grid, KDim, dtype=wpfloat, extend={KDim: 1})
    start_offset = time.time()
    _offset_k(f_k, out=f_offset, offset_provider={"Koff": KDim}, domain={KDim: (1, 11)})
    end_offset = time.time()

    f_k = random_field(grid, KDim, dtype=wpfloat, extend={KDim: 1})
    f_convert = zero_field(grid, KHalfDim, dtype=wpfloat)
    start_convert = time.time()
    _convert_khalfdim(
        f_k,
        out=f_convert,
        offset_provider={"KHalf2K": grid.get_offset_provider("KHalf2K")},
        domain={KHalfDim: (0, 10)},
    )
    end_convert = time.time()

    copy_time = end_copy - start_copy
    offset_time = end_offset - start_offset
    convert_time = end_convert - start_convert

    dict_times = {"convert_time": convert_time, "offset_time": offset_time, "copy_time": copy_time}
    print("max time: ", max(dict_times, key=dict_times.get))
    print("min_time: ", min(dict_times, key=dict_times.get))

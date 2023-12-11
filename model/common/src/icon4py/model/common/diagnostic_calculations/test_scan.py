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
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import (
    Field,
    float64,
    int32,
)
from gt4py.next.iterator.embedded import index_field, np_as_located_field
from gt4py.next import as_field

from icon4py.model.common.dimension import KDim

def test_scan_half():

    size = 10
    size_h = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=0.0,
    )
    def k_level_simple(
        state: float,
        input_var1: float,
        input_var2: float,
    ):
        return state + input_var1 * input_var2

    @scan_operator(
        axis=KDim,
        forward=True,
        init=0.0,
    )
    def k_level(
        state: float,
    ):
        return state + 1.0

    @field_operator
    def field_wrapper(
        input_var1: Field[[KDim], float],
        input_var2: Field[[KDim], float],
    ) -> Field[[KDim], float]:
        #output = k_level_simple(input_var1, input_var2)
        #output = input_var1 * input_var2
        output = k_level()
        return output

    @program
    def wrapper(
        input_var1: Field[[KDim], float],
        input_var2: Field[[KDim], float],
        output: Field[[KDim], float],
        vertical_start: int32,
        vertical_end: int32,
    ):
        field_wrapper(
            input_var1,
            input_var2,
            out=output,
            domain={
                KDim: (vertical_start, vertical_end),
            },
        )

    # gt4py version
    temp = np.ones((size,), dtype=float)
    temp[5:] = 2.0
    a = as_field((KDim,), np.arange(size, dtype=float))
    b = as_field((KDim,), temp)
    out = as_field((KDim,), np.zeros((size_h,), dtype=float))
    wrapper(a,b,out,0,10,offset_provider={})
    print(a.asnumpy())
    print(b.asnumpy())
    print(out.asnumpy())

    print("test_scan_half finish")

test_scan_half()

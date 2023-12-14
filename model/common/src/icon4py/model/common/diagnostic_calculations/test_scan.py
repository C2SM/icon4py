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
from gt4py.next import as_field

from icon4py.model.common.dimension import KDim, CellDim

def test_scan_half():

    size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=0.0
    )
    def k_level_simple(
        state: float,
        input_var1: float,
        input_var2: float,
    ):
        return state + input_var1 * input_var2

    @field_operator
    def field_wrapper(
        input_var1: Field[[KDim], float],
        input_var2: Field[[KDim], float],
    ) -> Field[[KDim], float]:
        output = k_level_simple(input_var1, input_var2)
        #output = input_var1 * input_var2
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
    out = as_field((KDim,), np.zeros((size,), dtype=float))
    wrapper(a,b,out,0,10,offset_provider={})
    print(a.asnumpy())
    print(b.asnumpy())
    print(out.asnumpy())

    # numpy version


    print("test_scan_half finish")

def test_slice():

    @field_operator
    def _surface(
        input_var1_slice1: Field[[CellDim], float],
        input_var1_slice2: Field[[CellDim], float],
        input_var1_slice3: Field[[CellDim], float],
    ):
        output = input_var1_slice1 + input_var1_slice2 + input_var1_slice3
        return output

    '''
    @field_operator
    def _computation(
        input_var1: Field[[CellDim, KDim], float],
        n_lev: int32,
    ):
        input_var1_slice1 = input_var1[0, n_lev - int32(1)]
        input_var1_slice2 = input_var1[:, n_lev - 2]
        input_var1_slice3 = input_var1[:, n_lev - 3]
        output = _surface(input_var1_slice1,input_var1_slice2,input_var1_slice3)
        return output

    @program
    def wrapper_NOTworked(
        input_var1: Field[[CellDim,KDim], float],
        output: Field[[CellDim], float],
        n_lev: int32,
        horizontal_start: int32,
        horizontal_end: int32,
        vertical_start: int32,
        vertical_end: int32,
    ):
        _computation(
            input_var1,
            n_lev,
            out=output,
            domain={
                CellDim: (horizontal_start, horizontal_end),
                KDim: (vertical_start, vertical_end),
            }
        )
    '''
    @program
    def wrapper_worked(
        input_var1_slice1: Field[[CellDim], float],
        input_var1_slice2: Field[[CellDim], float],
        input_var1_slice3: Field[[CellDim], float],
        output: Field[[CellDim], float],
        horizontal_start: int32,
        horizontal_end: int32,
    ):
        _surface(
            input_var1_slice1,
            input_var1_slice2,
            input_var1_slice3,
            out=output,
            domain={
                CellDim: (horizontal_start, horizontal_end),
            }
        )


    c_size = 10
    k_size = 5
    a_np = np.zeros((c_size, k_size), dtype=float)
    a_np[:, 0] = 1.0
    a_np[:, 1] = 2.0
    a_np[:, 2] = 3.0
    a = as_field((CellDim, KDim), a_np)
    a_1 = a[:, 0]
    a_2 = a[:, 1]
    a_3 = a[:, 2]
    out = as_field((CellDim,), np.zeros(c_size, dtype=float))
    #wrapper_NOTworked(a, k_size, 0, c_size, 0, k_size, out=out, offset_provider={})
    wrapper_worked(a_1, a_2, a_3, out, 0, c_size, offset_provider={})
    print(a_1.asnumpy())
    print(a_2.asnumpy())
    print(a_3.asnumpy())
    print(out.asnumpy())

test_slice()


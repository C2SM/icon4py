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

import numpy as np
from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import (
    Field,
    exp,
    float64,
    int32,
    log,
    neighbor_sum,
    sqrt,
)

# from icon4py.testutils.utils import to_icon4py_field, zero_field
from gt4py.next.iterator.embedded import index_field, np_as_located_field
from numpy import exp as numpy_exp
from numpy import log as numpy_log
from numpy import sqrt as numpy_sqrt

from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_math_utilities import gamma_fct
from icon4py.shared.mo_physical_constants import phy_const


class Constants(FrozenNamespace):

    two = 2.0
    four = 4.0
    GrConst_v1s = 0.50  # Exponent in the terminal velocity for snow
    GrConst_eta = 1.75e-5  # kinematic viscosity of air
    GrConst_ccsdep = (
        0.26 * gamma_fct((GrConst_v1s + 5.0) / 2.0) * numpy_sqrt(1.0 / GrConst_eta)
    )


myConst: Final = Constants()


def test_scan_sqrt():

    size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0),
    )
    def k_level_simple(state: float64, input_var1: float64) -> float64:
        return state + sqrt(input_var1) * myConst.GrConst_ccsdep

    @field_operator
    def wrapper(input_var1: Field[[KDim], float64]) -> Field[[KDim], float64]:
        output = k_level_simple(input_var1)
        return output

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(size, dtype=float64))
    out = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    wrapper(a, out=out, offset_provider={})
    print(out.array())

    # numpy version
    np_a = np.arange(size, dtype=float64)
    np_out = np.zeros(size, dtype=float64)
    np_out[0] = numpy_sqrt(np_a[0])
    for k in range(1, size):
        np_out[k] = np_out[k - 1] + numpy_sqrt(np_a[k]) * myConst.GrConst_ccsdep

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_sqrt finish")


def test_scan_log():

    size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0),
    )
    def k_level_simple(state: float64, input_var1: float64) -> float64:
        return state + log(input_var1) * myConst.GrConst_ccsdep

    @field_operator
    def wrapper(input_var1: Field[[KDim], float64]) -> Field[[KDim], float64]:
        output = k_level_simple(input_var1)
        return output

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(start=1, stop=size + 1, dtype=float64))
    out = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    wrapper(a, out=out, offset_provider={})
    print(out.array())

    # numpy version
    np_a = np.arange(start=1, stop=size + 1, dtype=float64)
    np_out = np.zeros(size, dtype=float64)
    np_out[0] = numpy_log(np_a[0])
    for k in range(1, size):
        np_out[k] = np_out[k - 1] + numpy_log(np_a[k]) * myConst.GrConst_ccsdep

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_log finish")


def test_scan_exp():

    size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0),
    )
    def k_level_simple(state: float64, input_var1: float64) -> float64:
        return state + exp(input_var1)

    @field_operator
    def wrapper(input_var1: Field[[KDim], float64]) -> Field[[KDim], float64]:
        output = k_level_simple(input_var1)
        return output

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(start=1, stop=size + 1, dtype=float64))
    out = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    wrapper(a, out=out, offset_provider={})
    print(out.array())

    # numpy version
    np_a = np.arange(start=1, stop=size + 1, dtype=float64)
    np_out = np.zeros(size, dtype=float64)
    np_out[0] = numpy_exp(np_a[0])
    for k in range(1, size):
        np_out[k] = np_out[k - 1] + numpy_exp(np_a[k])

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_exp finish")


def test_scan_float():

    size = 10
    threshold_level = 5.0

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0.0),
    )
    def k_level_simple(
        state: tuple[float64, float64], input_var1: float64, threshold_level: float64
    ) -> tuple[float64, float64]:
        if state[0] < threshold_level:
            return (state[0] + 1.0, state[1] + input_var1)
        else:
            return (state[0] + 1.0, state[1])

    @field_operator
    def wrapper(
        input_var1: Field[[KDim], float64],
        threshold_level: float64,
    ) -> Field[[KDim], float64]:
        reduncant, output = k_level_simple(input_var1, threshold_level)
        # input_var1_ = input_var1
        return output

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(size, dtype=float64))
    out = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    wrapper(a, threshold_level, out=out, offset_provider={})
    print(out.array())

    # numpy version
    np_a = np.arange(size, dtype=float64)
    np_out = np.zeros(size, dtype=float64)
    for k in range(1, size):
        if k < int(threshold_level):
            np_out[k] = np_out[k - 1] + np_a[k]
        else:
            np_out[k] = np_out[k - 1]

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_float finish")


def test_scan_integer():

    size = 10
    threshold_level = int32(5)

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0),
    )
    def k_level_simple(
        state: tuple[float64, int32], input_var1: float64, threshold_level: int32
    ) -> tuple[float64, int32]:
        (acm1, acm2) = state
        if acm2 < threshold_level:
            return (acm1 + input_var1, acm2 + int32(1))
        else:
            return (acm1, acm2 + int32(1))

    @field_operator
    def wrapper(
        input_var1: Field[[KDim], float64],
        threshold_level: int32,
    ) -> Field[[KDim], float64]:
        output, redundant = k_level_simple(input_var1, threshold_level)
        return output

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(size, dtype=float64))
    out = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    wrapper(a, threshold_level, out=out, offset_provider={})
    print(out.array())

    # numpy version
    np_a = np.arange(size, dtype=float64)
    np_out = np.zeros(size, dtype=float64)
    for k in range(1, size):
        if k < threshold_level:
            np_out[k] = np_out[k - 1] + np_a[k]
        else:
            np_out[k] = np_out[k - 1]

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_integer finish")


def test_k_level_1d2d():

    cell_size = 5
    k_size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=0.0,
    )
    def k_level_add(
        state: float64,
        input_var1: float64,
    ) -> float64:
        return state + input_var1

    @field_operator
    def wrapper(
        input_var1: Field[[CellDim, KDim], float64]
    ) -> Field[[CellDim], float64]:
        # output = neighbor_sum(input_var1, axis=KDim)
        # output = k_level_add(input_var1)
        # return (output)
        return input_var1[0]

    # numpy version
    np_cloud = np.ones((cell_size, k_size), dtype=float64)
    np_LWP = np.sum(np_cloud, axis=1)
    print(np_LWP)

    # gt4py version
    cloud = np_as_located_field(CellDim, KDim)(np.ones((cell_size, k_size)))
    LWP = np_as_located_field(CellDim)(np.zeros(cell_size))
    wrapper(cloud, out=LWP, offset_provider={})
    print(LWP.array())

    assert np.allclose(LWP.array(), np_LWP)

    print("test_k_level_2d3d finish")


def test_scan_multiple_output():

    size = 10
    threshold_level = int32(5)

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0.0, 0),
    )
    def k_scan(
        state: tuple[float64, float64, int32],
        input_var1: float64,
        input_var2: float64,
        threshold: int32,
    ) -> tuple[float64, float64, int32]:
        (acm1, acm2, acm3) = state
        if acm3 < threshold:
            return (acm1 + input_var1, acm2 + input_var2, acm3 + int32(1))
        else:
            return (acm1 + input_var1, acm2, acm3 + int32(1))

    @field_operator
    def wrapper_1output(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[KDim], float64],
        threshold: int32,
    ) -> Field[[KDim], float64]:
        output1, output2, redundant = k_scan(input_var1, input_var2, threshold)
        return output2

    @field_operator
    def wrapper_2output(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[KDim], float64],
        threshold: int32,
    ) -> tuple[Field[[KDim], float64], Field[[KDim], float64]]:
        output1, output2, redundant = k_scan(input_var1, input_var2, threshold)
        a = output1
        b = output2
        return a, b

    @field_operator
    def wrapper_3output(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[KDim], float64],
        threshold: int32,
    ) -> tuple[Field[[KDim], float64], Field[[KDim], float64], Field[[KDim], int32]]:
        (output1, output2, redundant) = k_scan(input_var1, input_var2, threshold)
        return (output1, output2, redundant)

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(size, dtype=float64))
    b = np_as_located_field(KDim)(np.ones((size,), dtype=float64))
    out1 = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    out2 = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    out3 = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    out4 = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    out5 = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    out6 = np_as_located_field(KDim)(np.zeros((size,), dtype=int32))
    wrapper_1output(a, b, threshold_level, out=out1, offset_provider={})
    wrapper_2output(a, b, threshold_level, out=(out2, out3), offset_provider={})
    wrapper_3output(a, b, threshold_level, out=(out4, out5, out6), offset_provider={})
    print(a.array())
    print(b.array())
    print(out1.array())
    print(out2.array())
    print(out3.array())

    # numpy version
    np_a = np.arange(size, dtype=float64)
    np_b = np.ones(size, dtype=float64)
    np_out1 = np.zeros(size, dtype=float64)
    np_out2 = np.zeros(size, dtype=float64)
    np_out3 = np.zeros(size, dtype=float64)
    np_out1[0] = np_out1[0] + np_b[0]
    np_out2[0] = np_out2[0] + np_b[0]
    for k in range(1, size):
        if k < threshold_level:
            np_out1[k] = np_out1[k - 1] + np_b[k]
            np_out2[k] = np_out2[k - 1] + np_b[k]
        else:
            np_out1[k] = np_out1[k - 1]
            np_out2[k] = np_out2[k - 1]
        np_out3[k] = np_out3[k - 1] + np_a[k]

    print(np_out1)
    assert np.allclose(out1.array(), np_out1)
    # assert np.allclose(out2.array(),np_out2)
    # assert np.allclose(out3.array(),np_out3)

    print("test_scan_multiple_output finish")

def test_program():

    size = 10
    threshold_level = int32(5)

    sequence1d = np.arange(size,dtype=float64)
    sequence2d = np.tile(sequence1d,(size,1))
    print(sequence2d)
    
    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0,0.0,0),
    )
    def k_scan(
        state: tuple[float64,float64,int32],
        input_var1: float64,
        input_var2: float64,
        input_var3: float64,
        threshold: int32
    ) -> tuple[float64,float64,int32]:
        (acm1, acm2, acm3) = state
        acm1 = input_var1 + input_var2
        if ( acm3 < threshold ):
            return (acm1, acm2 + input_var3, acm3 + int32(1))
        else:
            return (acm1, acm2, acm3 + int32(1))
    

    @field_operator
    def wrapper(
        input_var1: Field[[CellDim,KDim], float64],
        input_var2: Field[[CellDim,KDim], float64],
        input_var3: Field[[CellDim,KDim], float64],
        threshold: int32,
    ):
        output1, output2, redundant  = k_scan(input_var1,input_var2,input_var3,threshold)
        return (output1,output2)
    
    @program
    def program_wrapper(
        input_var1: Field[[CellDim,KDim], float64],
        input_var2: Field[[CellDim,KDim], float64],
        input_var3: Field[[CellDim,KDim], float64],
        threshold: int32,
        input_size: int32
    ):
        wrapper(
            input_var1,
            input_var2,
            input_var3,
            threshold,
            out=(
                input_var2,
                input_var3
            ),
            domain={CellDim: (0, input_size), KDim: (0, input_size)}
        )


    # gt4py version
    a = np_as_located_field(CellDim,KDim)(sequence2d)
    b = np_as_located_field(CellDim,KDim)(sequence2d)
    c = np_as_located_field(CellDim,KDim)(np.ones((size,size),dtype=float64))
    program_wrapper(a,b,c,threshold_level,size,offset_provider={})
    print("a: ", a.array())
    print("b: ", b.array())
    print("c: ", c.array())

    # numpy version
    np_a = sequence2d
    np_b = sequence2d
    np_c = np.ones((size,size),dtype=float64)
    np_out = np.zeros((size,size),dtype=float64)
    for i in range(0,size):
        np_out[i,0] = np_c[i,0]
        for k in range(1,size):
            if ( k < threshold_level ):
                np_out[i,k] = np_out[i,k-1] + np_c[i,k]
            else:
                np_out[i,k] = np_out[i,k-1]

    print(np_out)
    assert np.allclose(c.array(),np_out)

    print('test_program finish')

    
def test_k_level_ellipsis():

    cell_size = 5
    k_size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0.0, 0.0, 0.0),
    )
    def k_level_add(
        state: tuple[float64, ...],
        input_var1: float64,
        input_var2: float64,
    ) -> tuple[float64, ...]:
        return (state, state + 1.0, state + input_var1, state + input_var1 + input_var2)

    @field_operator
    def wrapper(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[KDim], float64],
    ) -> Field[[KDim], float64]:
        output1, output2, output3, output4 = k_level_add(input_var1, input_var2)
        return output4

    # numpy version
    np_cloud = np.ones(k_size, dtype=float64)
    np_ice = np.ones(k_size, dtype=float64)
    np_all = np.zeros(k_size, dtype=float64)
    for k in range(1, k_size):
        np_all[k] = np_all[k - 1] + np_cloud[k] + np_ice[k]
    print(np_all)

    # gt4py version
    cloud = np_as_located_field(KDim)(np.ones((k_size)))
    ice = np_as_located_field(KDim)(np.ones(k_size))
    out = np_as_located_field(KDim)(np.zeros(k_size))
    wrapper(cloud, out=out, offset_provider={})
    print(out.array())

    # assert np.allclose(LWP.array(),np_LWP)

    print("test_k_level_ellipsis finish")


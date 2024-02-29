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
    maximum,
    minimum,
    broadcast
)

# from icon4py.testutils.utils import to_icon4py_field, zero_field
from gt4py.next import as_field
from numpy import exp as numpy_exp
from numpy import log as numpy_log
from numpy import sqrt as numpy_sqrt

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.math.math_utilities import gamma_fct
from icon4py.model.common.mo_physical_constants import phy_const


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


def test_scan_dividebyzero():

    size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0),
    )
    def k_level_simple(state: float64, input_var1: float64) -> float64:
        if (state == 0.0):
            b = 1.0
        else:
            b = 1.0/input_var1
        return b

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
    np_out[0] = 1.0
    for k in range(1, size):
        np_out[k] = 1.0 / np_a[k]

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_dividebyzero finish")

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
        if ( state[0] < threshold_level ):
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
        if ( k < int(threshold_level) ):
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
        if ( acm2 < threshold_level) :
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
        if ( k < threshold_level ):
            np_out[k] = np_out[k - 1] + np_a[k]
        else:
            np_out[k] = np_out[k - 1]

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_integer finish")


def test_scan_bool():

    size = 10
    threshold_bool = True

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0),
    )
    def k_level_simple(
        state: tuple[float64, int32], input_var1: float64, threshold_bool: bool
    ) -> tuple[float64, int32]:
        (acm1, acm2) = state
        if ( threshold_bool ):
            return (acm1 + input_var1, acm2 + int32(1))
        else:
            return (acm1, acm2 + int32(1))

    @field_operator
    def wrapper(
        input_var1: Field[[KDim], float64],
        threshold_bool: bool,
    ) -> Field[[KDim], float64]:
        output, redundant = k_level_simple(input_var1, threshold_bool)
        return output

    # gt4py version
    a = np_as_located_field(KDim)(np.arange(size, dtype=float64))
    out = np_as_located_field(KDim)(np.zeros((size,), dtype=float64))
    wrapper(a, threshold_bool, out=out, offset_provider={})
    print(a.array())
    print(out.array())

    # numpy version
    np_a = np.arange(size, dtype=float64)
    np_out = np.zeros(size, dtype=float64)
    for k in range(1, size):
        if ( threshold_bool ):
            np_out[k] = np_out[k - 1] + np_a[k]
        else:
            np_out[k] = np_out[k - 1]

    print(np_out)
    assert np.allclose(out.array(), np_out)

    print("test_scan_bool finish")


def test_k_level_broadcast():
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
        input_var2: float64,
    ) -> float64:
        return state + input_var1 + input_var2

    @field_operator
    def wrapper_implicit_broadcast(
        input_var1: Field[[CellDim, KDim], float64],
        input_var2: Field[[CellDim], float64]
    ) -> Field[[CellDim,KDim], float64]:
        return k_level_add(input_var1, input_var2)


    @field_operator
    def wrapper_explicit_broadcast_needed(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[CellDim], float64]
    ) -> Field[[CellDim,KDim], float64]:
        return k_level_add(broadcast(input_var1, (CellDim, KDim)), input_var2)


    # gt4py version
    cloud2d = np_as_located_field(CellDim, KDim)(np.ones((cell_size, k_size), dtype=float64))
    cloud1d = np_as_located_field(KDim)(np.ones((k_size), dtype=float64))
    LWP1 = np_as_located_field(CellDim)(np.zeros(cell_size, dtype=float64))
    LWP2 = np_as_located_field(CellDim)(np.zeros(cell_size, dtype=float64))
    output1 = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size), dtype=float64))
    output2 = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size), dtype=float64))
    wrapper_implicit_broadcast(cloud2d, LWP1, out=output1, offset_provider={})
    wrapper_explicit_broadcast_needed(cloud1d, LWP2, out=output2, offset_provider={})
    print(output1.array())
    print(output2.array())

    # numpy version
    np_cloud = np.ones((cell_size, k_size), dtype=float64)
    np_LWP = np.sum(np_cloud, axis=1)
    print(np_LWP)

    # cannot compare, size is different. scan operator cannot output 1d array
    #assert np.allclose(output1.array(), np_LWP)
    #assert np.allclose(output2.array(), np_LWP)

    print("test_k_level_broadcast finish")


'''
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
'''

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
    np_out1[0] = np_out1[0] + np_b[0]
    np_out2[0] = np_out2[0] + np_a[0]
    for k in range(1, size):
        if k < threshold_level:
            np_out1[k] = np_out1[k - 1] + np_b[k]
        else:
            np_out1[k] = np_out1[k - 1]
        np_out2[k] = np_out2[k - 1] + np_a[k]

    print(np_out1)
    assert np.allclose(out1.array(), np_out1)
    assert np.allclose(out3.array(), np_out1)
    assert np.allclose(out5.array(), np_out1)

    print("test_scan_multiple_output finish")

def test_program():

    size = 10
    threshold_level = int32(5)

    sequence1d = np.arange(size,dtype=float64)
    sequence2d = np.tile(sequence1d,(size,1))
    print("original sequence: ", sequence2d)

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
    a = np_as_located_field(CellDim,KDim)(np.tile(sequence1d,(size,1)))
    b = np_as_located_field(CellDim,KDim)(np.tile(sequence1d,(size,1)))
    c = np_as_located_field(CellDim,KDim)(np.ones((size,size),dtype=float64))
    print("before a: ", a.array())
    print("before b: ", b.array())
    program_wrapper(a,b,c,threshold_level,size,offset_provider={})
    print("a: ", a.array())
    print("b: ", b.array())
    print("c: ", c.array())

    # numpy version
    np_a = np.tile(sequence1d,(size,1))
    np_b = np.tile(sequence1d,(size,1))
    np_c = np.ones((size,size),dtype=float64)
    np_out = np.zeros((size,size),dtype=float64)
    for i in range(0,size):
        np_out[i,0] = np_c[i,0]
        np_b[i,0] = np_b[i,0] + np_a[i,0]
        for k in range(1,size):
            np_b[i,k] = np_b[i,k] + np_a[i,k]
            if ( k < threshold_level ):
                np_out[i,k] = np_out[i,k-1] + np_c[i,k]
            else:
                np_out[i,k] = np_out[i,k-1]

    print(np_a)
    print(sequence2d)
    print(np_out)
    assert np.allclose(a.array(),np_a)
    assert np.allclose(b.array(),np_b)
    assert np.allclose(c.array(),np_out)

    print("--------------------------")
    np_a = sequence1d
    np_b = sequence1d
    print("test before a: ", np_a)
    print("test before b: ", np_b)
    for i in range(len(sequence1d)):
        np_b[i] = 0.0
    print("test after a: ", np_a)
    print("test after b: ", np_b)

    print('test_program finish')


def test_k_level_ellipsis():

    cell_size = 5
    k_size = 10

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(*(0.0,)*3,0.0),
        #init=(0.0,0.0,0.0,0.0),
    )
    def k_level_add(
        state: tuple[float64,float64, float64, float64],
        input_var1: float64,
        input_var2: float64,
    ) -> tuple[float64, float64, float64, float64]:
        return (state[0], state[1] + 1.0, state[2] + input_var1, state[3] + input_var1 + input_var2)

    @field_operator
    def wrapper(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[KDim], float64],
    ) -> tuple[Field[[KDim], float64],Field[[KDim], float64],Field[[KDim], float64],Field[[KDim], float64]]:
        output1, output2, output3, output4 = k_level_add(input_var1, input_var2)
        return (output1, output2, output3, output4)

    # numpy version
    np_cloud = np.ones(k_size, dtype=float64)
    np_ice = np.ones(k_size, dtype=float64)
    np_out1 = np.zeros(k_size, dtype=float64)
    np_out2 = np.zeros(k_size, dtype=float64)
    np_out3 = np.zeros(k_size, dtype=float64)
    np_out4 = np.zeros(k_size, dtype=float64)
    np_out2[0] = np_out2[0] + 1.0
    np_out3[0] = np_cloud[0]
    np_out4[0] = np_cloud[0] + np_ice[0]
    for k in range(1, k_size):
        np_out2[k] = np_out2[k - 1] + 1.0
        np_out3[k] = np_out3[k - 1] + np_cloud[k]
        np_out4[k] = np_out4[k - 1] + np_cloud[k] + np_ice[k]
    print(np_out4)

    # gt4py version
    cloud = np_as_located_field(KDim)(np.ones((k_size)))
    ice = np_as_located_field(KDim)(np.ones(k_size))
    out1 = np_as_located_field(KDim)(np.zeros(k_size))
    out2 = np_as_located_field(KDim)(np.zeros(k_size))
    out3 = np_as_located_field(KDim)(np.zeros(k_size))
    out4 = np_as_located_field(KDim)(np.zeros(k_size))
    wrapper(cloud,ice, out=(out1,out2,out3,out4), offset_provider={})
    print(out4.array())

    assert np.allclose(out1.array(),np_out1)
    assert np.allclose(out2.array(),np_out2)
    assert np.allclose(out3.array(),np_out3)
    assert np.allclose(out4.array(),np_out4)

    print("test_k_level_ellipsis finish")


def test_k_precFlux():

    cell_size = 5
    k_size = 10

    k_start = 3
    k_end = 9

    qmin = 1.e-15
    lred_depgrow = True
    ithermo_water = 0
    dt = 2.0

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
    )
    def cldtop(
        state: tuple[
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            int32
        ],
        dt: float64,
        dz: float64,
        temperature: float64,
        rho: float64,
        qv: float64,
        qc: float64,
        qi: float64,
        qs: float64,
        qg: float64,
        input_qmin: float64,
        input_lred_depgrow: bool,
        input_ithermo_water: int32,
        kstart: int32,
        kend: int32
    ) -> tuple[float64,float64,float64,float64,float64,float64,float64,float64,int32]:

        (
            temperature_kup,
            rho_kup,
            qv_kup,
            qc_kup,
            qi_kup,
            qs_kup,
            qg_kup,
            dist_cldtop,
            k_lev
        ) = state

        CLHv = phy_const.alv + (1850.0 - phy_const.clw) * (temperature - phy_const.tmelt) - phy_const.rv * temperature if (input_ithermo_water != int32(0)) else phy_const.alv
        CLHs = phy_const.als + (1850.0 - 2108.0) * (temperature - phy_const.tmelt) - phy_const.rv * temperature if (input_ithermo_water != int32(0)) else phy_const.als

        qvsw_kup = 610.78 * exp(17.269 * (temperature_kup - phy_const.tmelt) / (temperature_kup - 35.86)) / (rho_kup * phy_const.rv * temperature_kup)
        Cqvsi = 610.78 * exp(21.875 * (temperature - phy_const.tmelt) / (temperature - 7.66)) / (rho * phy_const.rv * temperature)
        nimix = 5.0 * exp(0.304 * (phy_const.tmelt - 250.15))

        if (input_lred_depgrow & (qc > input_qmin)):
            if ((k_lev > kstart) & (k_lev < kend)):

                # finalizing transfer rates in clouds and calculate depositional growth reduction
                # function called: Cnin_cooper = _fxna_cooper(temperature)
                Cnin_cooper = 5.0 * exp(0.304 * (phy_const.tmelt - temperature))
                Cnin_cooper = minimum(Cnin_cooper, 250.0e+3)
                Cfnuc = minimum(Cnin_cooper / nimix, 1.0)

                Cqcgk_1 = qi_kup + qs_kup + qg_kup

                # distance from cloud top
                if ((qv_kup + qc_kup < qvsw_kup) & (Cqcgk_1 < input_qmin)):
                    # upper cloud layer
                    dist_cldtop = 0.0  # reset distance to upper cloud layer
                else:
                    dist_cldtop = dist_cldtop + dz


                # with asymptotic behaviour dz -> 0 (xxx)
                #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
                #                             dist_cldtop(iv)/dist_cldtop_ref + &
                #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

                # without asymptotic behaviour dz -> 0
                reduce_dep = Cfnuc + (1.0 - Cfnuc) * (0.1 + dist_cldtop / 500.0)
                reduce_dep = minimum(reduce_dep, 1.0)

        Csdep = 3.367e-2
        Cidep = 1.3e-5
        Cslam = 1.0e10
        Cbsdep = 0.5 if (rho * qs > input_qmin) else 0.0

        local_qvsidiff = qv - Cqvsi

        Ssdep_v2s = 0.0
        reduce_dep = 1.0
        if ((qi > input_qmin) | (rho * qs > input_qmin) | (rho * qg > input_qmin)):

            if (temperature <= phy_const.tmelt):

                local_xfac = 1.0 + Cbsdep * exp(-0.25 * log(Cslam))
                Ssdep_v2s = Csdep * local_xfac * local_qvsidiff / (Cslam + 1.e-15) ** 2.0
                # FR new: depositional growth reduction
                if ((input_lred_depgrow) & (Ssdep_v2s > 0.0)):
                    Ssdep_v2s = Ssdep_v2s * reduce_dep

                if (qs <= 1.e-7):
                    Ssdep_v2s = minimum(Ssdep_v2s, 0.0)

        Cqvt = - Ssdep_v2s
        Cqst = Ssdep_v2s

        Ctt = phy_const.rcvd * CLHs * Cqst

        # Update variables and add qi to qrs for water loading
        qs = maximum( 0.0 , qs + Cqst * dt )
        temperature = temperature + Ctt * dt
        qv = maximum( 0.0 , qv + Cqvt * dt )

        k_lev = k_lev + 1
        return (temperature, rho, qv, qc, qi, qs, qg, dist_cldtop, k_lev)


    # numpy version
    np_cloud = np.ones(k_size, dtype=float64)
    np_ice = np.ones(k_size, dtype=float64)
    np_out1 = np.zeros(k_size, dtype=float64)
    np_out2 = np.zeros(k_size, dtype=float64)
    np_out3 = np.zeros(k_size, dtype=float64)
    np_out4 = np.zeros(k_size, dtype=float64)
    np_out2[0] = np_out2[0] + 1.0
    np_out3[0] = np_cloud[0]
    np_out4[0] = np_cloud[0] + np_ice[0]
    for k in range(1, k_size):
        np_out2[k] = np_out2[k - 1] + 1.0
        np_out3[k] = np_out3[k - 1] + np_cloud[k]
        np_out4[k] = np_out4[k - 1] + np_cloud[k] + np_ice[k]
    print(np_out4)

    # gt4py version
    dz = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=50.0, dtype=float64))
    temperature = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=250.0, dtype=float64))
    rho = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.0, dtype=float64))
    qv = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-3, dtype=float64))
    qc = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-4, dtype=float64))
    qi = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-5, dtype=float64))
    qs = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-5, dtype=float64))
    qg = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-5, dtype=float64))
    updated_temperature = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_rho = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qv = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qc = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qi = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qs = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qg = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    dist_cldtop = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    k_lev = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=int32))

    cldtop(dt,dz,temperature,rho,qv,qc,qi,qs,qg,qmin,lred_depgrow,ithermo_water,k_start,k_end, out=(updated_temperature,updated_rho,updated_qv,updated_qc,updated_qi,updated_qs,updated_qg,dist_cldtop,k_lev), offset_provider={})
    for i in range(cell_size):
        for k in range(k_size):
            print(qc.array()[i,k], qi.array()[i,k], qs.array()[i,k], dist_cldtop.array()[i,k])
        print()

    #assert np.allclose(out1.array(),np_out1)
    #assert np.allclose(out2.array(),np_out2)
    #assert np.allclose(out3.array(),np_out3)
    #assert np.allclose(out4.array(),np_out4)

    print("test_k_precFlux finish")

'''
class GraupelFunctionConstants(FrozenNamespace):

   GrFuncConst_n0s1  = 13.5 * 5.65e5 # parameter in N0S(T)
   GrFuncConst_n0s2  = -0.107        # parameter in N0S(T), Field et al
   GrFuncConst_mma   = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.312550,  0.000204,  0.003199, 0.000000, -0.015952)
   GrFuncConst_mmb   = (0.476221, -0.015896,  0.165977, 0.007468, -0.000141, 0.060366,  0.000079,  0.000594, 0.000000, -0.003577)

   GrFuncConst_thet  = 248.15 # temperature for het. nuc. of cloud ice

   GrFuncConst_ccau  = 4.0e-4     # autoconversion coefficient (cloud water to rain)
   GrFuncConst_cac   = 1.72       # (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8)
   GrFuncConst_kphi1 = 6.00e+02   # constant in phi-function for autoconversion
   GrFuncConst_kphi2 = 0.68e+00   # exponent in phi-function for autoconversion
   GrFuncConst_kphi3 = 5.00e-05   # exponent in phi-function for accretion
   GrFuncConst_kcau  = 9.44e+09   # kernel coeff for SB2001 autoconversion
   GrFuncConst_kcac  = 5.25e+00   # kernel coeff for SB2001 accretion
   GrFuncConst_cnue  = 2.00e+00   # gamma exponent for cloud distribution
   GrFuncConst_xstar = 2.60e-10   # separating mass between cloud and rain

   GrFuncConst_c1es    = 610.78                                               # = b1
   GrFuncConst_c2es    = GrFuncConst_c1es*phy_const.rd/phy_const.rv           #
   GrFuncConst_c3les   = 17.269                                               # = b2w
   GrFuncConst_c3ies   = 21.875                                               # = b2i
   GrFuncConst_c4les   = 35.86                                                # = b4w
   GrFuncConst_c4ies   = 7.66                                                 # = b4i
   GrFuncConst_c5les   = GrFuncConst_c3les*(phy_const.tmelt - GrFuncConst_c4les)      # = b234w
   GrFuncConst_c5ies   = GrFuncConst_c3ies*(phy_const.tmelt - GrFuncConst_c4ies)      # = b234i
   GrFuncConst_c5alvcp = GrFuncConst_c5les*phy_const.alv/phy_const.cpd            #
   GrFuncConst_c5alscp = GrFuncConst_c5ies*phy_const.als/phy_const.cpd            #
   GrFuncConst_alvdcp  = phy_const.alv/phy_const.cpd                          #
   GrFuncConst_alsdcp  = phy_const.als/phy_const.cpd                          #

   GrFuncConst_crim_g  = 4.43       # coefficient for graupel riming
   GrFuncConst_csg     = 0.5        # coefficient for snow-graupel conversion by riming
   GrFuncConst_cagg_g  = 2.46
   GrFuncConst_ciau    = 1.0e-3     # autoconversion coefficient (cloud ice to snow)
   GrFuncConst_msmin   = 3.0e-9     # initial mass of snow crystals
   GrFuncConst_cicri   = 1.72       # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
   GrFuncConst_crcri   = 1.24e-3    # (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
   GrFuncConst_asmel   = 2.95e3     # DIFF*LH_v*RHO/LHEAT
   GrFuncConst_tcrit   = 3339.5     # factor in calculation of critical temperature

   GrFuncConst_qc0 = 0.0            # param_qc0
   GrFuncConst_qi0 = 0.0            # param_qi0


graupel_funcConst : Final = GraupelFunctionConstants()

class GraupelGlobalConstants(FrozenNamespace):
    GrConst_qmin = 1.0e-15  # threshold for computations
    GrConst_eps = 1.0e-15  # small number

    GrConst_lstickeff = True  # switch for sticking coeff. (work from Guenther Zaengl)

    GrFuncConst_cagg_g = 2.46

    GrConst_rimexp_g = 0.94878

    GrConst_ceff_min = 0.01  # default: 0.075

    GrConst_ceff_fac = 3.5e-3  # Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency

graupel_const : Final = GraupelGlobalConstants()


def test_k_if():

    cell_size = 5
    k_size = 10

    k_start = 3
    k_end = 9

    qmin = 1.e-15
    lred_depgrow = True
    ithermo_water = 0
    dt = 2.0

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
    )
    def cldtop(
        state: tuple[
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            int32
        ],
        dt: float64,
        dz: float64,
        temperature: float64,
        rho: float64,
        qv: float64,
        qc: float64,
        qi: float64,
        qs: float64,
        qg: float64,
        input_qmin: float64,
        input_lred_depgrow: bool,
        input_ithermo_water: int32,
        kstart: int32,
        kend: int32
    ) -> tuple[float64,float64,float64,float64,float64,float64,float64,float64,int32]:

        (
            temperature_kup,
            rho_kup,
            qv_kup,
            qc_kup,
            qi_kup,
            qs_kup,
            qg_kup,
            dist_cldtop,
            k_lev
        ) = state

        CLHv = phy_const.alv + (1850.0 - phy_const.clw) * (temperature - phy_const.tmelt) - phy_const.rv * temperature if (input_ithermo_water != int32(0)) else phy_const.alv
        CLHs = phy_const.als + (1850.0 - 2108.0) * (temperature - phy_const.tmelt) - phy_const.rv * temperature if (input_ithermo_water != int32(0)) else phy_const.als

        qvsw_kup = 610.78 * exp(17.269 * (temperature_kup - phy_const.tmelt) / (temperature_kup - 35.86)) / (rho_kup * phy_const.rv * temperature_kup)
        Cqvsi = 610.78 * exp(21.875 * (temperature - phy_const.tmelt) / (temperature - 7.66)) / (rho * phy_const.rv * temperature)
        nimix = 5.0 * exp(0.304 * (phy_const.tmelt - 250.15))

        Cagg = 2.0

        Csdep = 3.367e-2
        Cidep = 1.3e-5
        Cslam = 1.0e10
        Cbsdep = 0.5 if (rho * qs > input_qmin) else 0.0

        Cdtr = 1.0 / dt

        rhoqg = rho * qg
        llqg = True if (rhoqg > graupel_const.GrConst_qmin) else False

        if (llqg):
            Clnrhoqg = log(rhoqg)
            Csgmax = rhoqg / rho * Cdtr
            if (qi + qc > graupel_const.GrConst_qmin):
                Celnrimexp_g = exp(graupel_const.GrConst_rimexp_g * Clnrhoqg)
            Celn6qgk = exp(0.6 * Clnrhoqg)

        Saggs_i2s = 0.0
        Saggg_i2g = 0.0
        if ((temperature <= phy_const.tmelt) & (qi > input_qmin)):

            # Change in sticking efficiency needed in case of cloud ice sedimentation
            # (based on Guenther Zaengls work)
            if (graupel_const.GrConst_lstickeff):
                local_eff = minimum(exp(0.09 * (temperature - phy_const.tmelt)), 1.0)
                local_eff = maximum(local_eff, graupel_const.GrConst_ceff_min)
                local_eff = maximum(local_eff, graupel_const.GrConst_ceff_fac * (
                        temperature - graupel_const.GrConst_tmin_iceautoconv))
            else:  # original sticking efficiency of cloud ice
                local_eff = minimum(exp(0.09 * (temperature - phy_const.tmelt)), 1.0)
                local_eff = maximum(local_eff, 0.2)

            Saggs_i2s = local_eff * qi * Cagg * exp(graupel_const.GrConst_ccsaxp * log(Cslam))
            Saggg_i2g = local_eff * qi * graupel_funcConst.GrFuncConst_cagg_g * Celnrimexp_g
            # Siaut_i2s = local_eff * graupel_funcConst.GrFuncConst_ciau * maximum( qi - graupel_funcConst.GrFuncConst_qi0 , 0.0 )

            Sicri_i2g = graupel_funcConst.GrFuncConst_cicri * qi * Celn7o8qrk
            if (qs > 1.e-7):
                Srcri_r2g = graupel_funcConst.GrFuncConst_crcri * (qi / Cmi) * Celn13o8qrk

        Cqvt = - Ssdep_v2s
        Cqst = Ssdep_v2s

        Ctt = phy_const.rcvd * CLHs * Cqst

        # Update variables and add qi to qrs for water loading
        qs = maximum( 0.0 , qs + Cqst * dt )
        temperature = temperature + Ctt * dt
        qv = maximum( 0.0 , qv + Cqvt * dt )

        k_lev = k_lev + 1
        return (temperature, rho, qv, qc, qi, qs, qg, dist_cldtop, k_lev)


    # numpy version
    np_cloud = np.ones(k_size, dtype=float64)
    np_ice = np.ones(k_size, dtype=float64)
    np_out1 = np.zeros(k_size, dtype=float64)
    np_out2 = np.zeros(k_size, dtype=float64)
    np_out3 = np.zeros(k_size, dtype=float64)
    np_out4 = np.zeros(k_size, dtype=float64)
    np_out2[0] = np_out2[0] + 1.0
    np_out3[0] = np_cloud[0]
    np_out4[0] = np_cloud[0] + np_ice[0]
    for k in range(1, k_size):
        np_out2[k] = np_out2[k - 1] + 1.0
        np_out3[k] = np_out3[k - 1] + np_cloud[k]
        np_out4[k] = np_out4[k - 1] + np_cloud[k] + np_ice[k]
    print(np_out4)

    # gt4py version
    dz = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=50.0, dtype=float64))
    temperature = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=250.0, dtype=float64))
    rho = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.0, dtype=float64))
    qv = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-3, dtype=float64))
    qc = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-4, dtype=float64))
    qi = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-5, dtype=float64))
    qs = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-5, dtype=float64))
    qg = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.e-5, dtype=float64))
    updated_temperature = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_rho = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qv = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qc = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qi = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qs = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    updated_qg = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    dist_cldtop = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    k_lev = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=int32))

    cldtop(dt,dz,temperature,rho,qv,qc,qi,qs,qg,qmin,lred_depgrow,ithermo_water,k_start,k_end, out=(updated_temperature,updated_rho,updated_qv,updated_qc,updated_qi,updated_qs,updated_qg,dist_cldtop,k_lev), offset_provider={})
    for i in range(cell_size):
        for k in range(k_size):
            print(qc.array()[i,k], qi.array()[i,k], qs.array()[i,k], dist_cldtop.array()[i,k])
        print()

    #assert np.allclose(out1.array(),np_out1)
    #assert np.allclose(out2.array(),np_out2)
    #assert np.allclose(out3.array(),np_out3)
    #assert np.allclose(out4.array(),np_out4)

    print("test_k_if finish")
'''

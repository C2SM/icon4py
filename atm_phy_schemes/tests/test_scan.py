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
    broadcast
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

    @field_operator
    def velocity_precFlux(
        input_TV: float64,
        V_intg_factor: float64,
        V_intg_exp: float64,
        input_rhoq: float64,
        input_q: float64,
        input_q_kup: float64,
        input_rho_kup: float64,
        rho_factor: float64,
        rho_factor_kup: float64,
        V_intg_exp_1o2: float64,
        input_V_sedi_min: float64,
        intercept_param: float64,
        input_is_top: bool,
        input_is_surface: bool,
        is_intercept_param: bool
    ) -> tuple[float64,float64]:

        TV = 0.0
        precFlux = 0.0
        terminal_velocity = V_intg_factor * exp (V_intg_exp * log (input_rhoq)) * rho_factor
        # Prevent terminal fall speed of snow from being zero at the surface level
        if ( (input_V_sedi_min != 0.0) & input_is_surface ):

            terminal_velocity = maximum( terminal_velocity, input_V_sedi_min )

        #if ( input_rhoq > graupel_const.GrConst_qmin ):
        #   precFlux = input_rhoq * terminal_velocity
        TV = terminal_velocity

        return (TV, precFlux)

    @scan_operator(
        axis=KDim,
        forward=True,
        init=(0.0,0.0),
    )
    def k_level_add(
        state: tuple[float64,float64],
        input_var1: float64,
        input_var2: float64,
    ) -> tuple[float64, float64]:
        (vnew_kup, q_kup) = state
        vnew, precflux = velocity_precFlux(vnew_kup,1.0,1.0,input_var1,input_var2,q_kup,1.0,1.0,1.0,1.0,1.0,1.0,True,True,True)
        return (vnew, q_kup)

    @field_operator
    def wrapper(
        input_var1: Field[[KDim], float64],
        input_var2: Field[[KDim], float64],
    ) -> tuple[Field[[KDim], float64],Field[[KDim], float64]]:
        output1, output2 = k_level_add(input_var1, input_var2)
        return (output1, output2)

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
    wrapper(cloud,ice, out=(out1,out2), offset_provider={})
    print(out4.array())

    #assert np.allclose(out1.array(),np_out1)
    #assert np.allclose(out2.array(),np_out2)
    #assert np.allclose(out3.array(),np_out3)
    #assert np.allclose(out4.array(),np_out4)

    print("test_k_precFlux finish")


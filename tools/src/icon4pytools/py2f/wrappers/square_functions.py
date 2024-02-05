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

# flake8: noqa D100, D103

import numpy as np
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Dimension, Field, float64
from gt4py.next.iterator.embedded import np_as_located_field


CellDim = Dimension("Cell")
KDim = Dimension("K")
VDim = Dimension("Vertex")
EDim = Dimension("Edge")


@field_operator
def _multiply_fields(
    a: Field[[CellDim, KDim], float64], b: Field[[CellDim, KDim], float64]
) -> Field[[CellDim, KDim], float64]:
    return a * b


@program
def multiply_fields(
    a: Field[[CellDim, KDim], float64],
    b: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    _multiply_fields(a, b, out=result)


@program
def square_fields(a: Field[[CellDim, KDim], float64], result: Field[[CellDim, KDim], float64]):
    _multiply_fields(a, a, out=result)


def square_d(input: np.ndarray, output: np.ndarray, func):
    input_field = np_as_located_field(CellDim, KDim)(input)
    output_field = np_as_located_field(CellDim, KDim)(output)
    func(input_field, input_field, output_field, offset_provider={})


def square_return(input_ar: np.ndarray) -> np.array:
    a_field = np_as_located_field(CellDim, KDim)(input_ar)
    result = np_as_located_field(CellDim, KDim)(np.zeros(input_ar.shape, dtype=float64))
    multiply_fields(a_field, a_field, result, offset_provider={})
    return np.asarray(result)


def square_output_param(input_ar: np.ndarray, output_ar: np.ndarray):
    input_field = np_as_located_field(CellDim, KDim)(input_ar)
    output_field = np_as_located_field(CellDim, KDim)(output_ar)
    multiply_fields(input_field, input_field, output_field, offset_provider={})


def square(state):
    """pass input and output arrays inside **state"""
    a = state["input"]
    res = state["output"]
    assert res.shape == a.shape
    square_output_param(a, res)

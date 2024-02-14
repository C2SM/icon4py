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
import string

import pytest
from gt4py.next.type_system.type_specifications import ScalarKind
from icon4py.model.common.dimension import CellDim, KDim

from icon4pytools.py2fgen.codegen import (
    CffiPlugin,
    CHeaderGenerator,
    F90InterfaceGenerator,
    Func,
    FuncParameter,
    as_f90_value,
)


field_2d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[CellDim, KDim],
    py_type_hint="Field[CellDim, KDim], float64]",
)
field_1d = FuncParameter(
    name="name", d_type=ScalarKind.FLOAT32, dimensions=[KDim], py_type_hint="Field[KDim], float64]"
)

simple_type = FuncParameter(
    name="name", d_type=ScalarKind.FLOAT32, dimensions=[], py_type_hint="int32"
)


@pytest.mark.parametrize(
    ("param", "expected"), ((simple_type, "value,"), (field_2d, ""), (field_1d, ""))
)
def test_as_target(param, expected):
    assert expected == as_f90_value(param)


# TODO: adapt test to new functions
@pytest.mark.skip
@pytest.mark.parametrize(("lang", "expected"), (("C", "*"), ("F", "dimension(:,:),")))
def test_field_extension_2d(lang, expected):
    pass


# TODO: adapt test to new functions
@pytest.mark.skip
@pytest.mark.parametrize(("lang", "expected"), (("C", "*"), ("F", "dimension(:),")))
def test_field_extension_1d(lang, expected):
    pass


# TODO: adapt test to new functions
@pytest.mark.skip
@pytest.mark.parametrize("lang", ("C", "F"))
def test_is_field_simple_type(lang):
    pass


foo = Func(
    name="foo",
    args=[
        FuncParameter(name="one", d_type=ScalarKind.INT32, dimensions=[], py_type_hint="int32"),
        FuncParameter(
            name="two",
            d_type=ScalarKind.FLOAT64,
            dimensions=[CellDim, KDim],
            py_type_hint="Field[CellDim, KDim], float64]",
        ),
    ],
    is_gt4py_program=False,
)

bar = Func(
    name="bar",
    args=[
        FuncParameter(
            name="one",
            d_type=ScalarKind.FLOAT32,
            dimensions=[
                CellDim,
                KDim,
            ],
            py_type_hint="Field[CellDim, KDim], float64]",
        ),
        FuncParameter(name="two", d_type=ScalarKind.INT32, dimensions=[], py_type_hint="int32"),
    ],
    is_gt4py_program=False,
)


def test_cheader_generation_for_single_function():
    plugin = CffiPlugin(module_name="libtest", function=foo, imports=["import foo"])

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void foo_wrapper(int one, double* two, int n_cell, int n_k);"


def test_cheader_for_pointer_args():
    plugin = CffiPlugin(module_name="libtest", function=bar, imports=["import bar"])

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void bar_wrapper(float* one, int two, int n_cell, int n_k);"


def compare_ignore_whitespace(s1: str, s2: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    return s1.translate(no_whitespace) == s2.translate(no_whitespace)


def test_fortran_interface():
    plugin = CffiPlugin(module_name="libtest", function=foo, imports=["import foo"])
    interface = F90InterfaceGenerator.apply(plugin)
    expected = """
    module libtest_plugin
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
        subroutine foo_wrapper(one, &
                       two, &
                       n_cell, &
                       n_k) bind(c, name='foo_wrapper')
            use, intrinsic :: iso_c_binding
            integer(c_int), value, target :: n_cell
            integer(c_int), value, target :: n_k
            integer(c_int), value, target :: one
            real(c_double), dimension(:, :), target :: two(n_cell, n_k)
        end subroutine foo_wrapper
    end interface
    end module
    """
    assert compare_ignore_whitespace(interface, expected)

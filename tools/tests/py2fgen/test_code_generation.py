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
    as_field,
)


field_2d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[CellDim, KDim],
)
field_1d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[KDim],
)

simple_type = FuncParameter(name="name", d_type=ScalarKind.FLOAT32, dimensions=[])


@pytest.mark.parametrize(
    ("param", "expected"), ((simple_type, "value,"), (field_2d, ""), (field_1d, ""))
)
def test_as_target(param, expected):
    assert expected == as_f90_value(param)


@pytest.mark.parametrize(("lang", "expected"), (("C", "*"), ("F", "dimension(:,:),")))
def test_field_extension_2d(lang, expected):
    assert as_field(field_2d, lang) == expected


@pytest.mark.parametrize(("lang", "expected"), (("C", "*"), ("F", "dimension(:),")))
def test_field_extension_1d(lang, expected):
    assert as_field(field_1d, lang) == expected


@pytest.mark.parametrize("lang", ("C", "F"))
def test_is_field_simple_type(lang):
    assert as_field(simple_type, lang) == ""


foo = Func(
    name="foo",
    args=[
        FuncParameter(name="one", d_type=ScalarKind.INT32, dimensions=[]),
        FuncParameter(name="two", d_type=ScalarKind.FLOAT64, dimensions=[CellDim, KDim]),
    ],
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
        ),
        FuncParameter(name="two", d_type=ScalarKind.INT32, dimensions=[]),
    ],
)


def test_cheader_generation_for_single_function():
    functions = [foo]
    plugin = CffiPlugin(name="libtest", functions=functions)

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void foo(int one, double* two);\n"


def test_cheader_for_pointer_args():
    functions = [bar]
    plugin = CffiPlugin(name="libtest", functions=functions)

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void bar(float* one, int two);\n"


def compare_ignore_whitespace(s1: str, s2: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    return s1.translate(no_whitespace) == s2.translate(no_whitespace)


def test_c_header_with_several_functions():
    functions = [bar, foo]
    plugin = CffiPlugin(name="libtest", functions=functions)
    header = CHeaderGenerator.apply(plugin)
    assert (
        header
        == """extern void bar(float* one, int two);\nextern void foo(int one, double* two);\n"""
    )


def test_fortran_interface():
    functions = [foo]
    plugin = CffiPlugin(name="libtest", functions=functions)
    interface = F90InterfaceGenerator.apply(plugin)
    expected = """
    module libtest
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
        subroutine foo(one, &
                       two) bind(c, name='foo')
            use, intrinsic :: iso_c_binding
            integer(c_int), value, target :: one
            real(c_double), dimension(:, :), target :: two(n_cell, n_k)
        end subroutine foo
    end interface
    end module
    """
    assert compare_ignore_whitespace(interface, expected)

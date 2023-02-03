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
from functional.type_system.type_specifications import ScalarKind

from icon4py.diffusion.wrapper.binding import (
    CffiPlugin,
    CHeaderGenerator,
    DimensionType,
    F90InterfaceGenerator,
    Func,
    FuncParameter,
    field_extension,
)


fieldParam2d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[DimensionType(name="K", length=13), DimensionType(name="J", length=13)],
)
fieldParam1d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[DimensionType(name="K", length=13)],
)

simpleType = FuncParameter(name="name", d_type=ScalarKind.FLOAT32, dimensions=[])


@pytest.mark.parametrize(("lang", "expected"), (("C", "*"), ("F", "(:,:)")))
def test_field_extension_2d(lang, expected):
    assert field_extension(fieldParam2d, lang) == expected


@pytest.mark.parametrize(("lang", "expected"), (("C", "*"), ("F", "(:)")))
def test_field_extension_1d(lang, expected):
    assert field_extension(fieldParam1d, lang) == expected


@pytest.mark.parametrize("lang", ("C", "F"))
def test_is_field_simple_type(lang):
    assert field_extension(simpleType, lang) == ""


foo = Func(
    name="foo",
    args=[
        FuncParameter(name="one", d_type=ScalarKind.INT32, dimensions=[]),
        FuncParameter(name="two", d_type=ScalarKind.FLOAT64, dimensions=[]),
    ],
)

bar = Func(
    name="bar",
    args=[
        FuncParameter(
            name="one",
            d_type=ScalarKind.FLOAT32,
            dimensions=[
                DimensionType(name="KDim", length=10),
                DimensionType(name="VDim", length=50000),
            ],
        ),
        FuncParameter(name="two", d_type=ScalarKind.INT32, dimensions=[]),
    ],
)


def test_cheader_generation_for_single_function():
    functions = [foo]
    plugin = CffiPlugin(name="libtest", functions=functions)

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void foo(int one, double two);\n"


def test_cheader_for_pointer_args():
    functions = [bar]
    plugin = CffiPlugin(name="libtest", functions=functions)

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void bar(float* one, int two);\n"


def compare_ignore_whitespace(s1: str, s2: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    return s1.translate(no_whitespace) == s2.translate(no_whitespace)


def test_cheader_with_several_functions():
    functions = [bar, foo]
    plugin = CffiPlugin(name="libtest", functions=functions)
    header = CHeaderGenerator.apply(plugin)
    assert (
        header
        == """extern void bar(float* one, int two);\nextern void foo(int one, double two);\n"""
    )


def test_fortran_interface():
    functions = [foo]
    plugin = CffiPlugin(name="libtest", functions=functions)
    interface = F90InterfaceGenerator.apply(plugin)
    expteced = """
    module libtest
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
        subroutine foo(one, two) bind(c, name='foo')
        use iso_c_binding
        integer(c_int), intent(inout):: one
        real(c_double), intent(inout):: two
        end subroutine foo
    end interface
    end module
    """
    assert compare_ignore_whitespace(interface, expteced)

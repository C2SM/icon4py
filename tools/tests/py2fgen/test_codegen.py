# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import string

import pytest
from gt4py.next.type_system.type_specifications import ScalarKind

from icon4py.model.common import dimension as dims
from icon4py.tools.py2fgen.generate import (
    generate_c_header,
    generate_f90_interface,
    generate_python_wrapper,
)
from icon4py.tools.py2fgen.template import (
    CffiPlugin,
    CHeaderGenerator,
    Func,
    FuncParameter,
    as_f90_value,
)


field_2d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[dims.CellDim, dims.KDim],
)
field_1d = FuncParameter(
    name="name",
    d_type=ScalarKind.FLOAT32,
    dimensions=[dims.KDim],
)

simple_type = FuncParameter(name="name", d_type=ScalarKind.FLOAT32, dimensions=[])


@pytest.mark.parametrize(
    ("param", "expected"), ((simple_type, "value"), (field_2d, None), (field_1d, None))
)
def test_as_target(param, expected):
    assert expected == as_f90_value(param)


foo = Func(
    name="foo",
    args=[
        FuncParameter(name="one", d_type=ScalarKind.INT32, dimensions=[]),
        FuncParameter(
            name="two",
            d_type=ScalarKind.FLOAT64,
            dimensions=[dims.CellDim, dims.KDim],
        ),
    ],
)

bar = Func(
    name="bar",
    args=[
        FuncParameter(
            name="one",
            d_type=ScalarKind.FLOAT32,
            dimensions=[
                dims.CellDim,
                dims.KDim,
            ],
        ),
        FuncParameter(name="two", d_type=ScalarKind.INT32, dimensions=[]),
    ],
)


def test_cheader_generation_for_single_function():
    plugin = CffiPlugin(module_name="libtest", plugin_name="libtest_plugin", functions=[foo])

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern int foo_wrapper(int one, double* two, int two_size_0, int two_size_1);"


def test_cheader_for_pointer_args():
    plugin = CffiPlugin(module_name="libtest", plugin_name="libtest_plugin", functions=[bar])

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern int bar_wrapper(float* one, int one_size_0, int one_size_1, int two);"


def compare_ignore_whitespace(actual: str, expected: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    if actual.translate(no_whitespace) != expected.translate(no_whitespace):
        print("Expected:")
        print(expected)
        print("Actual:")
        print("------------------------------")
        print(actual)
        print("------------------------------")
        return False
    return True


@pytest.fixture
def dummy_plugin():
    return CffiPlugin(
        module_name="libtest",
        plugin_name="libtest_plugin",
        functions=[foo, bar],
    )


def test_fortran_interface(dummy_plugin):
    interface = generate_f90_interface(dummy_plugin)
    expected = """
module libtest_plugin
   use, intrinsic :: iso_c_binding
   implicit none

   public :: foo

   public :: bar

   interface

      function foo_wrapper(one, &
                           two, &
                           two_size_0, &
                           two_size_1) bind(c, name="foo_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         integer(c_int), value, target :: one

         type(c_ptr), value, target :: two

         integer(c_int), value :: two_size_0

         integer(c_int), value :: two_size_1

      end function foo_wrapper

      function bar_wrapper(one, &
                           one_size_0, &
                           one_size_1, &
                           two) bind(c, name="bar_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: one

         integer(c_int), value :: one_size_0

         integer(c_int), value :: one_size_1

         integer(c_int), value, target :: two

      end function bar_wrapper

   end interface

contains

   subroutine foo(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      integer(c_int), value, target :: one

      real(c_double), dimension(:, :), target :: two

      integer(c_int) :: two_size_0

      integer(c_int) :: two_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(two)

      two_size_0 = SIZE(two, 1)
      two_size_1 = SIZE(two, 2)

      rc = foo_wrapper(one=one, &
                       two=c_loc(two), &
                       two_size_0=two_size_0, &
                       two_size_1=two_size_1)
      !$acc end host_data
   end subroutine foo

   subroutine bar(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      real(c_float), dimension(:, :), target :: one

      integer(c_int), value, target :: two

      integer(c_int) :: one_size_0

      integer(c_int) :: one_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(one)

      one_size_0 = SIZE(one, 1)
      one_size_1 = SIZE(one, 2)

      rc = bar_wrapper(one=c_loc(one), &
                       one_size_0=one_size_0, &
                       one_size_1=one_size_1, &
                       two=two)
      !$acc end host_data
   end subroutine bar

end module
"""
    assert compare_ignore_whitespace(interface, expected)


def test_python_wrapper(dummy_plugin):
    interface = generate_python_wrapper(dummy_plugin, "GPU", False, profile=False)
    expected = """# imports for generated wrapper code
import logging

from libtest_plugin import ffi
import cupy as cp
import gt4py.next as gtx
from gt4py.next.type_system import type_specifications as ts
from icon4py.tools.py2fgen.settings import config
from icon4py.tools.py2fgen import wrapper_utils

xp = config.array_ns

# logger setup
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.ERROR, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
logging.info(cp.show_config())

# embedded function imports
from libtest import foo
from libtest import bar


Cell = gtx.Dimension("Cell", kind=gtx.DimensionKind.HORIZONTAL)
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


@ffi.def_extern()
def foo_wrapper(one, two, two_size_0, two_size_1):
    try:

        # Convert ptr to GT4Py fields

        two = wrapper_utils.as_field(
            ffi, xp, two, ts.ScalarKind.FLOAT64, {Cell: two_size_0, K: two_size_1}, False
        )

        foo(one, two)

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def bar_wrapper(one, one_size_0, one_size_1, two):
    try:

        # Convert ptr to GT4Py fields

        one = wrapper_utils.as_field(
            ffi, xp, one, ts.ScalarKind.FLOAT32, {Cell: one_size_0, K: one_size_1}, False
        )

        bar(one, two)

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0

"""
    assert compare_ignore_whitespace(interface, expected)


def test_c_header(dummy_plugin):
    interface = generate_c_header(dummy_plugin)
    expected = """
    extern int foo_wrapper(int one, double *two, int two_size_0, int two_size_1);
    extern int bar_wrapper(float *one, int one_size_0, int one_size_1, int two);
    """
    assert compare_ignore_whitespace(interface, expected)

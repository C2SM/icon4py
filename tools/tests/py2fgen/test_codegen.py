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
    ("param", "expected"), ((simple_type, "value,"), (field_2d, ""), (field_1d, ""))
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
    assert header == "extern int foo_wrapper(int one, double* two, int n_Cell, int n_K);"


def test_cheader_for_pointer_args():
    plugin = CffiPlugin(module_name="libtest", plugin_name="libtest_plugin", functions=[bar])

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern int bar_wrapper(float* one, int two, int n_Cell, int n_K);"


def compare_ignore_whitespace(s1: str, s2: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    return s1.translate(no_whitespace) == s2.translate(no_whitespace)


@pytest.fixture
def dummy_plugin():
    return CffiPlugin(
        module_name="libtest",
        plugin_name="libtest_plugin",
        functions=[foo, bar],
    )


def test_fortran_interface(dummy_plugin):
    interface = generate_f90_interface(dummy_plugin, limited_area=True)
    expected = """
    module libtest_plugin
   use, intrinsic :: iso_c_binding
   implicit none

   public :: foo

   public :: bar

   interface

      function foo_wrapper(one, &
                           two, &
                           n_Cell, &
                           n_K) bind(c, name="foo_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_K

         integer(c_int) :: rc  ! Stores the return code

         integer(c_int), value, target :: one

         real(c_double), dimension(*), target :: two

      end function foo_wrapper

      function bar_wrapper(one, &
                           two, &
                           n_Cell, &
                           n_K) bind(c, name="bar_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_K

         integer(c_int) :: rc  ! Stores the return code

         real(c_float), dimension(*), target :: one

         integer(c_int), value, target :: two

      end function bar_wrapper

   end interface

contains

   subroutine foo(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_Cell

      integer(c_int) :: n_K

      integer(c_int), value, target :: one

      real(c_double), dimension(:, :), target :: two

      integer(c_int) :: rc  ! Stores the return code

      !$ACC host_data use_device( &
      !$ACC two &
      !$ACC )

      n_Cell = SIZE(two, 1)

      n_K = SIZE(two, 2)

      rc = foo_wrapper(one, &
                       two, &
                       n_Cell, &
                       n_K)

      !$acc end host_data
   end subroutine foo

   subroutine bar(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_Cell

      integer(c_int) :: n_K

      real(c_float), dimension(:, :), target :: one

      integer(c_int), value, target :: two

      integer(c_int) :: rc  ! Stores the return code

      !$ACC host_data use_device( &
      !$ACC one &
      !$ACC )

      n_Cell = SIZE(one, 1)

      n_K = SIZE(one, 2)

      rc = bar_wrapper(one, &
                       two, &
                       n_Cell, &
                       n_K)

      !$acc end host_data
   end subroutine bar

end module
"""
    assert compare_ignore_whitespace(interface, expected)


def test_python_wrapper(dummy_plugin):
    interface = generate_python_wrapper(
        dummy_plugin, "GPU", False, limited_area=True, profile=False
    )
    expected = """
# imports for generated wrapper code
import logging

from libtest_plugin import ffi
import cupy as cp
from gt4py.next.iterator.embedded import np_as_located_field
from icon4py.tools.py2fgen.settings import config
from icon4py.tools.py2fgen import wrapper_utils
from icon4py.model.common import dimension as dims

# logger setup
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.ERROR, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
logging.info(cp.show_config())

# embedded function imports
from libtest import foo
from libtest import bar


@ffi.def_extern()
def foo_wrapper(one, two, n_Cell, n_K):
    try:

        # Unpack pointers into Ndarrays

        two = wrapper_utils.unpack_gpu(ffi, two, n_Cell, n_K)

        # Allocate GT4Py Fields

        two = np_as_located_field(dims.CellDim, dims.KDim)(two)

        foo(one, two)

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def bar_wrapper(one, two, n_Cell, n_K):
    try:

        # Unpack pointers into Ndarrays

        one = wrapper_utils.unpack_gpu(ffi, one, n_Cell, n_K)

        # Allocate GT4Py Fields

        one = np_as_located_field(dims.CellDim, dims.KDim)(one)

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
    extern int foo_wrapper(int one, double *two, int n_Cell, int n_K);
    extern int bar_wrapper(float *one, int two, int n_Cell, int n_K);
    """
    assert compare_ignore_whitespace(interface, expected)

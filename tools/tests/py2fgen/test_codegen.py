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

from icon4pytools.py2fgen.generate import (
    generate_c_header,
    generate_f90_interface,
    generate_python_wrapper,
)
from icon4pytools.py2fgen.template import (
    CffiPlugin,
    CHeaderGenerator,
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
    plugin = CffiPlugin(
        module_name="libtest", plugin_name="libtest_plugin", functions=[foo], imports=["import foo"]
    )

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern int foo_wrapper(int one, double* two, int n_Cell, int n_K);"


def test_cheader_for_pointer_args():
    plugin = CffiPlugin(
        module_name="libtest", plugin_name="libtest_plugin", functions=[bar], imports=["import bar"]
    )

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
        imports=["import foo_module_x\nimport bar_module_y"],
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
    interface = generate_python_wrapper(dummy_plugin, "GPU", False, limited_area=True)
    expected = '''
# imports for generated wrapper code
import logging
import math
from libtest_plugin import ffi
import numpy as np
import cupy as cp
from numpy.typing import NDArray
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.ffront.fbuiltins import int32
from icon4py.model.common.settings import xp

# logger setup
log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.ERROR,
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(cp.show_config())

# embedded module imports
import foo_module_x
import bar_module_y

# embedded function imports
from libtest import foo
from libtest import bar

def unpack_gpu(ptr, *sizes: int):
    """
    Converts a C pointer into a CuPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations that require in-place modification of GPU data,
    enabling changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ptr (cffi.CData): A CFFI pointer to GPU memory allocated by OpenACC, representing
                          the starting address of the data. This pointer must correspond to
                          a contiguous block of memory whose total size matches the product
                          of the specified dimensions.
        *sizes (int): Variable length argument list specifying the dimensions of the array.
                      These sizes determine the shape of the resulting CuPy array.

    Returns:
        cp.ndarray: A CuPy array that provides a direct view of the data pointed to by `ptr`.
                    This array shares the underlying data with the original Fortran code, allowing
                    modifications made through the array to affect the original data.
    """
    if not sizes:
        raise ValueError("Sizes must be provided to determine the array shape.")

    length = math.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    dtype_map = {
        "int": cp.int32,
        "double": cp.float64,
    }
    dtype = dtype_map.get(c_type, None)
    if dtype is None:
        raise ValueError(f"Unsupported C data type: {c_type}")

    itemsize = ffi.sizeof(c_type)
    total_size = length * itemsize

    # cupy array from OpenACC device pointer
    current_device = cp.cuda.Device()
    ptr_val = int(ffi.cast("uintptr_t", ptr))
    mem = cp.cuda.UnownedMemory(ptr_val, total_size, owner=ptr, device_id=current_device.id)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    arr = cp.ndarray(shape=sizes, dtype=dtype, memptr=memptr, order="F")
    return arr

def int_array_to_bool_array(int_array: NDArray) -> NDArray:
    """
    Converts a NumPy array of integers to a boolean array.
    In the input array, 0 represents False, and any non-zero value (1 or -1) represents True.

    Args:
        int_array: A NumPy array of integers.

    Returns:
        A NumPy array of booleans.
    """
    bool_array = int_array != 0
    return bool_array

@ffi.def_extern()
def foo_wrapper(one: int32, two: Field[CellDim, KDim], float64], n_Cell: int32, n_K: int32):
    try:
        # Unpack pointers into Ndarrays
        two = unpack_gpu(two, n_Cell, n_K)

        # Allocate GT4Py Fields
        two = np_as_located_field(CellDim, KDim)(two)

        foo(one, two)

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0

@ffi.def_extern()
def bar_wrapper(one: Field[CellDim, KDim], float64], two: int32, n_Cell: int32, n_K: int32):
    try:
        # Unpack pointers into Ndarrays
        one = unpack_gpu(one, n_Cell, n_K)

        # Allocate GT4Py Fields
        one = np_as_located_field(CellDim, KDim)(one)

        bar(one, two)

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0
    '''
    assert compare_ignore_whitespace(interface, expected)


def test_c_header(dummy_plugin):
    interface = generate_c_header(dummy_plugin)
    expected = """
    extern int foo_wrapper(int one, double *two, int n_Cell, int n_K);
    extern int bar_wrapper(float *one, int two, int n_Cell, int n_K);
    """
    assert compare_ignore_whitespace(interface, expected)

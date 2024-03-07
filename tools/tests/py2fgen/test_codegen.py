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
        module_name="libtest", plugin_name="libtest_plugin", function=[foo], imports=["import foo"]
    )

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void foo_wrapper(int one, double* two, int n_Cell, int n_K);"


def test_cheader_for_pointer_args():
    plugin = CffiPlugin(
        module_name="libtest", plugin_name="libtest_plugin", function=[bar], imports=["import bar"]
    )

    header = CHeaderGenerator.apply(plugin)
    assert header == "extern void bar_wrapper(float* one, int two, int n_Cell, int n_K);"


def compare_ignore_whitespace(s1: str, s2: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    return s1.translate(no_whitespace) == s2.translate(no_whitespace)


@pytest.fixture
def dummy_plugin():
    return CffiPlugin(
        module_name="libtest",
        plugin_name="libtest_plugin",
        function=[foo, bar],
        imports=["import foo_module_x\nimport bar_module_y"],
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

      subroutine foo_wrapper(one, &
                             two, &
                             n_Cell, &
                             n_K) bind(c, name="foo_wrapper")
         import :: c_int, c_double

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_K

         integer(c_int), value, target :: one

         real(c_double), dimension(*), target :: two

      end subroutine foo_wrapper

      subroutine bar_wrapper(one, &
                             two, &
                             n_Cell, &
                             n_K) bind(c, name="bar_wrapper")
         import :: c_int, c_double

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_K

         real(c_float), dimension(*), target :: one

         integer(c_int), value, target :: two

      end subroutine bar_wrapper

   end interface

contains

   subroutine foo(one, &
                  two)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_Cell

      integer(c_int) :: n_K

      integer(c_int), value, target :: one

      real(c_double), dimension(:, :), target :: two

      n_Cell = SIZE(two, 1)

      n_K = SIZE(two, 2)

      call foo_wrapper(one, &
                       two, &
                       n_Cell, &
                       n_K)

   end subroutine foo

   subroutine bar(one, &
                  two)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_Cell

      integer(c_int) :: n_K

      real(c_float), dimension(:, :), target :: one

      integer(c_int), value, target :: two

      n_Cell = SIZE(one, 1)

      n_K = SIZE(one, 2)

      call bar_wrapper(one, &
                       two, &
                       n_Cell, &
                       n_K)

   end subroutine bar

end module
    """
    assert compare_ignore_whitespace(interface, expected)


def test_python_wrapper(dummy_plugin):
    interface = generate_python_wrapper(dummy_plugin, None, False)
    expected = '''
# necessary imports for generated code to work
from libtest_plugin import ffi
import numpy as np
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next import as_field
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_gpu
from gt4py.next.program_processors.runners.roundtrip import backend as run_roundtrip
from icon4py.model.common.grid.simple import SimpleGrid

# all other imports from the module from which the function is being wrapped
import foo_module_x
import bar_module_y

# We need a grid to pass offset providers
grid = SimpleGrid()

from libtest import foo
from libtest import bar


def unpack(ptr, *sizes: int) -> np.ndarray:
    """
    Converts a C pointer pointing to data in Fortran (column-major) order into a NumPy array.

    This function facilitates the handling of numerical data shared between C (or Fortran) and Python,
    especially when the data originates from Fortran routines that use column-major storage order.
    It creates a NumPy array that directly views the data pointed to by `ptr`, without copying, and reshapes
    it according to the specified dimensions. The resulting NumPy array uses Fortran order ('F') to preserve
    the original data layout.

    Args:
        ptr (CData): A CFFI pointer to the beginning of the data array. This pointer should reference
            a contiguous block of memory whose total size matches the product of the specified dimensions.
        *sizes (int): Variable length argument list representing the dimensions of the array. The product
            of these sizes should match the total number of elements in the data block pointed to by `ptr`.

    Returns:
        np.ndarray: A NumPy array view of the data pointed to by `ptr`. The array will have the shape
        specified by `sizes` and the data type (`dtype`) corresponding to the C data type of `ptr`.
        The array is created with Fortran order to match the original column-major data layout.

    Note:
        The function does not perform any copying of the data. Modifications to the resulting NumPy array
        will affect the original data pointed to by `ptr`. Ensure that the lifetime of the data pointed to
        by `ptr` extends beyond the use of the returned NumPy array to prevent data corruption or access
        violations.
    """
    length = np.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    # Map C data types to NumPy dtypes
    dtype_map: dict[str, np.dtype] = {
        "int": np.dtype(np.int32),
        "double": np.dtype(np.float64),
    }
    dtype = dtype_map.get(c_type, np.dtype(c_type))

    # TODO(samkellerhals): see if we can fix type issue
    # Create a NumPy array from the buffer, specifying the Fortran order
    arr = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), dtype=dtype).reshape(  # type: ignore
        sizes, order="F"
    )
    return arr

@ffi.def_extern()
def foo_wrapper(one: int32, two: Field[CellDim, KDim], float64], n_Cell: int32, n_K: int32):
    # Unpack pointers into Ndarrays
    two = unpack(two, n_Cell, n_K)

    # Allocate GT4Py Fields
    two = np_as_located_field(CellDim, KDim)(two)

    foo(one, two)

@ffi.def_extern()
def bar_wrapper(one: Field[CellDim, KDim], float64], two: int32, n_Cell: int32, n_K: int32):
    # Unpack pointers into Ndarrays
    one = unpack(one, n_Cell, n_K)

    # Allocate GT4Py Fields
    one = np_as_located_field(CellDim, KDim)(one)

    bar(one, two)
    '''
    assert compare_ignore_whitespace(interface, expected)


def test_c_header(dummy_plugin):
    interface = generate_c_header(dummy_plugin)
    expected = """
    extern void foo_wrapper(int one, double *two, int n_Cell, int n_K);
    extern void bar_wrapper(float *one, int two, int n_Cell, int n_K);
    """
    assert compare_ignore_whitespace(interface, expected)

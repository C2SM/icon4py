# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import string

import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen._codegen import (
    CffiPlugin,
    CHeaderGenerator,
    Func,
    as_f90_value,
    generate_c_header,
    generate_f90_interface,
    generate_python_wrapper,
)


field_2d = py2fgen.ArrayParamDescriptor(
    rank=2,
    dtype=py2fgen.FLOAT32,
    device=py2fgen.DeviceType.MAYBE_DEVICE,
    is_optional=False,
)

field_1d = py2fgen.ArrayParamDescriptor(
    rank=1,
    dtype=py2fgen.FLOAT32,
    device=py2fgen.DeviceType.MAYBE_DEVICE,
    is_optional=False,
)


simple_type = py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)


@pytest.mark.parametrize(
    ("param", "expected"), ((simple_type, "value"), (field_2d, None), (field_1d, None))
)
def test_as_target(param, expected):
    assert expected == as_f90_value(param)


foo = Func(
    name="foo",
    args={
        "one": py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        "two": py2fgen.ArrayParamDescriptor(
            rank=2,
            dtype=py2fgen.FLOAT64,
            device=py2fgen.DeviceType.MAYBE_DEVICE,
            is_optional=False,
        ),
    },
)

bar = Func(
    name="bar",
    args={
        "one": py2fgen.ArrayParamDescriptor(
            rank=2,
            dtype=py2fgen.FLOAT32,
            device=py2fgen.DeviceType.MAYBE_DEVICE,
            is_optional=False,
        ),
        "two": py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
    },
)


def test_cheader_generation_for_single_function():
    plugin = CffiPlugin(module_name="libtest", plugin_name="libtest_plugin", functions=[foo])

    header = CHeaderGenerator.apply(plugin)
    assert (
        header
        == "extern int foo_wrapper(int one, double* two, int two_size_0, int two_size_1, int on_gpu);"
    )


def test_cheader_for_pointer_args():
    plugin = CffiPlugin(module_name="libtest", plugin_name="libtest_plugin", functions=[bar])

    header = CHeaderGenerator.apply(plugin)
    assert (
        header
        == "extern int bar_wrapper(float* one, int one_size_0, int one_size_1, int two, int on_gpu);"
    )


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
                           two_size_1, &
                           on_gpu) bind(c, name="foo_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         integer(c_int), value, target :: one

         type(c_ptr), value, target :: two

         integer(c_int), value :: two_size_0

         integer(c_int), value :: two_size_1

         logical(c_int), value :: on_gpu

      end function foo_wrapper

      function bar_wrapper(one, &
                           one_size_0, &
                           one_size_1, &
                           two, &
                           on_gpu) bind(c, name="bar_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: one

         integer(c_int), value :: one_size_0

         integer(c_int), value :: one_size_1

         integer(c_int), value, target :: two

         logical(c_int), value :: on_gpu

      end function bar_wrapper

   end interface

contains

   subroutine foo(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      integer(c_int), value, target :: one

      real(c_double), dimension(:, :), target :: two

      logical(c_int) :: on_gpu

      integer(c_int) :: two_size_0

      integer(c_int) :: two_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(two)

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      two_size_0 = SIZE(two, 1)
      two_size_1 = SIZE(two, 2)

      rc = foo_wrapper(one=one, &
                       two=c_loc(two), &
                       two_size_0=two_size_0, &
                       two_size_1=two_size_1, &
                       on_gpu=on_gpu)
      !$acc end host_data
   end subroutine foo

   subroutine bar(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      real(c_float), dimension(:, :), target :: one

      integer(c_int), value, target :: two

      logical(c_int) :: on_gpu

      integer(c_int) :: one_size_0

      integer(c_int) :: one_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(one)

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      one_size_0 = SIZE(one, 1)
      one_size_1 = SIZE(one, 2)

      rc = bar_wrapper(one=c_loc(one), &
                       one_size_0=one_size_0, &
                       one_size_1=one_size_1, &
                       two=two, &
                       on_gpu=on_gpu)
      !$acc end host_data
   end subroutine bar

end module
"""
    assert compare_ignore_whitespace(interface, expected)


def test_python_wrapper(dummy_plugin):
    interface = generate_python_wrapper(dummy_plugin)
    expected = """import logging
from libtest_plugin import ffi
from icon4py.tools.py2fgen import utils, runtime_config, _runtime, _definitions

if __debug__:
    logger = logging.getLogger(__name__)
    log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, runtime_config.LOG_LEVEL),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# embedded function imports
from libtest import foo
from libtest import bar


@ffi.def_extern()
def foo_wrapper(one, two, two_size_0, two_size_1, on_gpu):
    try:
        if __debug__:
            logger.info("Python execution of foo started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayInfos

        two = (
            two,
            (
                two_size_0,
                two_size_1,
            ),
            on_gpu,
            False,
        )

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "foo constructing `ArrayInfos` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            meta = {}
        else:
            meta = None
        foo(
            ffi=ffi,
            meta=meta,
            one=one,
            two=two,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "foo convert time: %s"
                    % str(meta["convert_end_time"] - meta["convert_start_time"])
                )
                logger.info("foo execution time: %s" % str(func_end_time - func_start_time))

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                msg = "shape of two after computation = %s" % str(
                    two.shape if two is not None else "None"
                )
                logger.debug(msg)
                msg = "two after computation: %s" % str(
                    utils.as_array(ffi, two, _definitions.FLOAT64) if two is not None else "None"
                )
                logger.debug(msg)

        if __debug__:
            logger.info("Python execution of foo completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def bar_wrapper(one, one_size_0, one_size_1, two, on_gpu):
    try:
        if __debug__:
            logger.info("Python execution of bar started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayInfos

        one = (
            one,
            (
                one_size_0,
                one_size_1,
            ),
            on_gpu,
            False,
        )

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "bar constructing `ArrayInfos` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            meta = {}
        else:
            meta = None
        bar(
            ffi=ffi,
            meta=meta,
            one=one,
            two=two,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "bar convert time: %s"
                    % str(meta["convert_end_time"] - meta["convert_start_time"])
                )
                logger.info("bar execution time: %s" % str(func_end_time - func_start_time))

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                msg = "shape of one after computation = %s" % str(
                    one.shape if one is not None else "None"
                )
                logger.debug(msg)
                msg = "one after computation: %s" % str(
                    utils.as_array(ffi, one, _definitions.FLOAT32) if one is not None else "None"
                )
                logger.debug(msg)

        if __debug__:
            logger.info("Python execution of bar completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0

"""
    assert compare_ignore_whitespace(interface, expected)


def test_c_header(dummy_plugin):
    interface = generate_c_header(dummy_plugin)
    expected = """
    extern int foo_wrapper(int one, double *two, int two_size_0, int two_size_1, int on_gpu);
    extern int bar_wrapper(float *one, int one_size_0, int one_size_1, int two, int on_gpu);
    """
    assert compare_ignore_whitespace(interface, expected)

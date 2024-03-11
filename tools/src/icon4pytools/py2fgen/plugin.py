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
import logging
from pathlib import Path

import cffi
import numpy as np
from cffi import FFI
from numpy.typing import NDArray

from icon4pytools.common.logger import setup_logger


ffi = FFI()

logger = setup_logger(__name__)


def unpack(ptr, *sizes: int) -> NDArray:
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

    # Create a NumPy array from the buffer, specifying the Fortran order
    arr = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), dtype=dtype).reshape(  # type: ignore
        sizes, order="F"
    )
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


def generate_and_compile_cffi_plugin(
    plugin_name: str, c_header: str, python_wrapper: str, build_path: Path
) -> None:
    """
    Create and compile a CFFI plugin.

    This function generates a C shared library and Fortran interface for Python functions
    to be exposed in the {plugin_name} module. It creates a linkable C library named
    'lib{plugin_name}.so' in the specified build directory.

    Args:
        plugin_name: Name of the plugin.
        c_header: C header signatures for the Python functions.
        python_wrapper: Python code wrapping the original function to be exposed.
        build_path: Path to the build directory.
    """
    try:
        header_file_path = write_c_header(build_path, plugin_name, c_header)
        compile_cffi_plugin(
            builder=configure_cffi_builder(c_header, plugin_name, header_file_path),
            python_wrapper=python_wrapper,
            build_path=str(build_path),
            plugin_name=plugin_name,
        )
    except Exception as e:
        logging.error(f"Error generating and compiling CFFI plugin: {e}")
        raise


def write_c_header(build_path: Path, plugin_name: str, c_header: str) -> Path:
    """Write the C header file to the specified path."""
    c_header_file = plugin_name + ".h"
    header_file_path = build_path / c_header_file
    with open(header_file_path, "w") as f:
        f.write(c_header)
    return header_file_path


def configure_cffi_builder(c_header: str, plugin_name: str, header_file_path: Path) -> cffi.FFI:
    """Configure and returns a CFFI FFI builder instance."""
    builder = cffi.FFI()
    builder.embedding_api(c_header)
    builder.set_source(plugin_name, f'#include "{header_file_path.name}"')
    return builder


def compile_cffi_plugin(
    builder: cffi.FFI, python_wrapper: str, build_path: str, plugin_name: str
) -> None:
    """Compile the CFFI plugin with the given configuration."""
    logger.info("Compiling CFFI dynamic library...")
    builder.embedding_init_code(python_wrapper)
    builder.compile(tmpdir=build_path, target=f"lib{plugin_name}.*", verbose=True)

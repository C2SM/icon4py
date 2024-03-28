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
import cupy as cp  # type: ignore
import numpy as np
from cffi import FFI
from numpy.typing import NDArray

from icon4pytools.common.logger import setup_logger


ffi = FFI()  # needed for unpack and unpack_gpu functions

logger = setup_logger(__name__)


def unpack(ptr, *sizes: int) -> NDArray:
    """
    Converts a C pointer into a NumPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations requiring in-place modification of CPU data, enabling
    changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ptr (CData): A CFFI pointer to the beginning of the data array in CPU memory. This pointer
                     should reference a contiguous block of memory whose total size matches the product
                     of the specified dimensions.
        *sizes (int): Variable length argument list specifying the dimensions of the array.
                      These sizes determine the shape of the resulting NumPy array.

    Returns:
        np.ndarray: A NumPy array that provides a direct view of the data pointed to by `ptr`.
                    This array shares the underlying data with the original Fortran code, allowing
                    modifications made through the array to affect the original data.
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


def unpack_gpu(ptr, *sizes: int) -> cp.ndarray:
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

    length = np.prod(sizes)
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

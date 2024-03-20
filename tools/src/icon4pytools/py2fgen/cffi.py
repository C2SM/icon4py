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

from icon4pytools.common.logger import setup_logger


ffi = FFI()

logger = setup_logger(__name__)


def unpack(ptr, *sizes: int) -> np.ndarray:
    """
    Unpacks an n-dimensional Fortran (column-major) array into a NumPy array (row-major).

    This function is particularly useful when interfacing with C code that returns arrays
    in column-major order (Fortran order), but you want to work with the array in row-major
    order (NumPy's default order).

    Args:
        ptr: A C pointer to the field.
        *sizes: Size arguments representing the length of each dimension of the array in
            row-major order. The length of this argument must match the number of dimensions
            of the array.

    Returns:
        A NumPy array with shape specified by the sizes and dtype determined by the ctype
        (C data type) of the pointer.
    """
    length = np.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    # special casing different types
    if c_type == "int":
        dtype = np.int32
    else:
        dtype = np.dtype(c_type)  # type: ignore

    arr = np.frombuffer(  # type: ignore
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=dtype,
        count=-1,
        offset=0,
    ).reshape(sizes)
    return arr


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

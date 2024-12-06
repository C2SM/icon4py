# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import typing
from pathlib import Path
from typing import Tuple

import cffi
import gt4py.next as gtx
import numpy as np
from cffi import FFI
from icon4py.model.common.settings import xp
from numpy.typing import NDArray

from icon4pytools.common.logger import setup_logger

ffi = FFI()  # needed for unpack and unpack_gpu functions

logger = setup_logger(__name__)


def unpack(ptr: cffi.api.FFI.CData, *sizes: int) -> NDArray:
    """
    Converts a C pointer into a NumPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations requiring in-place modification of CPU data, enabling
    changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ptr (ffi.CData): A CFFI pointer to the beginning of the data array in CPU memory. This pointer
                     should reference a contiguous block of memory whose total size matches the product
                     of the specified dimensions.
        *sizes (int): Variable length argument list specifying the dimensions of the array.
                      These sizes determine the shape of the resulting NumPy array.

    Returns:
        np.ndarray: A NumPy array that provides a direct view of the data pointed to by `ptr`.
                    This array shares the underlying data with the original Fortran code, allowing
                    modifications made through the array to affect the original data.
    """
    length = math.prod(sizes)
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


def unpack_gpu(ptr: cffi.api.FFI.CData, *sizes: int):
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
        "int": xp.int32,
        "double": xp.float64,
    }
    dtype = dtype_map.get(c_type, None)
    if dtype is None:
        raise ValueError(f"Unsupported C data type: {c_type}")

    itemsize = ffi.sizeof(c_type)
    total_size = length * itemsize

    # cupy array from OpenACC device pointer
    current_device = xp.cuda.Device()
    ptr_val = int(ffi.cast("uintptr_t", ptr))
    mem = xp.cuda.UnownedMemory(ptr_val, total_size, owner=ptr, device_id=current_device.id)
    memptr = xp.cuda.MemoryPointer(mem, 0)
    arr = xp.ndarray(shape=sizes, dtype=dtype, memptr=memptr, order="F")
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


def unpack_and_cache_pointer(
    pointer: cffi.api.FFI.CData,
    key: Tuple[str, Tuple[int, ...]],
    sizes: list[int],
    gt_dims: list[gtx.Dimension],
    dtype: xp.dtype,
    is_uninitialized: bool,
    is_bool: bool,
    backend: str,
    cache: dict,
    xp: xp,
):
    """
    Unpacks and caches a Fortran pointer, retrieving a cached version if available.

    This function checks if a field corresponding to the given `key` exists in the
    cache. If the field does not exist, it unpacks the pointer data, optionally
    converts it to a boolean array, allocates the field with specified dimensions, and stores
    it in the cache. If the field exists, it is directly retrieved from the cache.

    Args:
        pointer: The raw pointer or data to be unpacked into the field.
        key: A unique identifier for the field, typically a tuple of the field name
            and its shape.
        sizes: A list of dimension sizes for unpacking the pointer.
        gt_dims: A list of GT4Py dimensions for creating a gt4py field.
        dtype: The data type of the field (e.g., `numpy.float32`).
        is_uninitialized (bool): Whether the field is uninitialized and should be filled with ones.
        is_bool (bool): Whether the field contains boolean data and needs conversion.
        backend (str): The backend in use, e.g., `"GPU"` or `"CPU"`, to determine unpacking logic.
        cache (dict): The dictionary storing cached fields, keyed by `field_key`.
        xp (module): The numerical library in use, such as `numpy` or `cupy`.

    Returns:
        Any: The allocated and cached field corresponding to `field_key`.

    Raises:
        KeyError: If a required field key or dimension is missing during the caching process.

    Example:
        ```python
        cached_field = get_cached_field(
            field_key=("temperature", ("n_Cell", "n_K")),
            pointer=temp_pointer,
            sizes=[n_Cell, n_K],
            gt_dims=[dims.CellDim, dims.KDim],
            dtype=np.float64,
            is_uninitialized=False,
            is_bool=False,
            backend="CPU",
            field_dict=allocated_fields,
            xp=np,
        )
        ```
    """
    if key not in cache:
        if is_uninitialized:
            # in these instances the field is filled with garbage values as it is not used by ICON.
            unpacked = xp.ones((1,) * len(sizes), dtype=dtype, order="F")
        else:
            unpacked = unpack_gpu(pointer, *sizes) if backend == "GPU" else unpack(pointer, *sizes)
            
        if is_bool:
            unpacked = int_array_to_bool_array(unpacked)

        cache[key] = gtx.np_as_located_field(*gt_dims)(unpacked)

    return cache[key]


def generate_and_compile_cffi_plugin(
    plugin_name: str, c_header: str, python_wrapper: str, build_path: Path, backend: str
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
        backend: Backend used by the generated C shared library.
    """
    try:
        header_file_path = write_c_header(build_path, plugin_name, c_header)
        compile_cffi_plugin(
            builder=configure_cffi_builder(c_header, plugin_name, header_file_path),
            python_wrapper=python_wrapper,
            build_path=str(build_path),
            plugin_name=plugin_name,
            backend=backend,
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
    builder: cffi.FFI, python_wrapper: str, build_path: str, plugin_name: str, backend: str
) -> None:
    """Compile the CFFI plugin with the given configuration."""
    logger.info("Compiling CFFI dynamic library...")
    builder.embedding_init_code(python_wrapper)
    builder.compile(tmpdir=build_path, target=f"lib{plugin_name}_{backend.lower()}.*", verbose=True)

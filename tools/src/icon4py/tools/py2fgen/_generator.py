# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from collections.abc import Callable
from pathlib import Path

import cffi

from icon4py.tools.py2fgen import _codegen, _utils


def get_cffi_description(functions: list[Callable], library_name: str) -> _codegen.BindingsLibrary:
    """Build the CFFI plugin description from already-imported callables.

    Each callable's import location is derived from ``__module__``, so
    callers can mix functions from different source modules in a single
    library. The generated Python wrapper imports each function from its
    own definition module.
    """
    return _codegen.BindingsLibrary(
        library_name=library_name,
        functions=[_get_function_descriptor(f) for f in functions],
    )


def _get_function_descriptor(fun: Callable) -> _codegen.Func:
    if not hasattr(fun, "param_descriptors"):
        raise TypeError("Cannot parse function, did you forget to decorate it with '@export'?")
    return _codegen.Func(name=fun.__name__, module_name=fun.__module__, args=fun.param_descriptors)


def configure_cffi_builder(
    library_name: str,
    c_header: str,
    python_wrapper: str,
    build_path: Path,
    rpath: str,
) -> cffi.FFI:
    """Write the C header, configure and return a CFFI FFI builder ready for compilation or code emission."""
    header_file_path = build_path / f"{library_name}.h"
    with header_file_path.open("w") as f:
        f.write(c_header)

    builder = cffi.FFI()
    extra_link_args = [f"-Wl,-rpath={rpath}"] if rpath else []
    builder.embedding_api(c_header)
    builder.set_source(
        library_name, f'#include "{header_file_path.name}"', extra_link_args=extra_link_args
    )
    builder.embedding_init_code(python_wrapper)
    return builder


def generate_and_compile_cffi_plugin(
    library_name: str,
    c_header: str,
    python_wrapper: str,
    build_path: Path,
    rpath: str = _utils.get_prefix_lib_path(),
) -> None:
    """
    Create and compile a CFFI plugin.

    Generates a C shared library for Python functions to be exposed in the
    {library_name} module. Creates a linkable C library named
    'lib{library_name}.so' in the specified build directory.

    Args:
        library_name: Name of the plugin.
        c_header: C header signatures for the Python functions.
        python_wrapper: Python code wrapping the original function to be exposed.
        build_path: Path to the build directory.
        rpath: Runtime library search path to embed in the shared library.
    """
    try:
        builder = configure_cffi_builder(library_name, c_header, python_wrapper, build_path, rpath)
        builder.compile(tmpdir=str(build_path), target=f"lib{library_name}.*", verbose=True)
    except Exception as e:
        logging.error(f"Error generating and compiling CFFI plugin: {e}")
        raise


def generate_cffi_source(
    library_name: str,
    c_header: str,
    python_wrapper: str,
    build_path: Path,
    rpath: str = _utils.get_prefix_lib_path(),
) -> None:
    """Generate the C source file and header without compiling."""
    try:
        builder = configure_cffi_builder(library_name, c_header, python_wrapper, build_path, rpath)
        builder.emit_c_code(str(build_path / f"{library_name}.c"))
    except Exception as e:
        logging.error(f"Error generating CFFI C source: {e}")
        raise

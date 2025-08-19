# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import logging
from collections.abc import Callable
from pathlib import Path

import cffi

from icon4py.tools.py2fgen import _codegen, _utils


def get_cffi_description(
    module_name: str, functions: list[str], library_name: str
) -> _codegen.BindingsLibrary:
    # TODO(havogt): instead of a list of function names, we could just export all
    # exportable functions of the module(functions with the `param_descriptors` attribute).
    module = importlib.import_module(module_name)
    parsed_functions = [_get_function_descriptor(getattr(module, f)) for f in functions]
    return _codegen.BindingsLibrary(
        module_name=module_name,
        library_name=library_name,
        functions=parsed_functions,
    )


def _get_function_descriptor(fun: Callable) -> _codegen.Func:
    if not hasattr(fun, "param_descriptors"):
        raise TypeError("Cannot parse function, did you forget to decorate it with '@export'?")
    return _codegen.Func(name=fun.__name__, args=fun.param_descriptors)


def generate_and_compile_cffi_plugin(
    library_name: str,
    c_header: str,
    python_wrapper: str,
    build_path: Path,
    rpath: str = _utils.get_prefix_lib_path(),
) -> None:
    """
    Create and compile a CFFI plugin.

    This function generates a C shared library and Fortran interface for Python functions
    to be exposed in the {library_name} module. It creates a linkable C library named
    'lib{library_name}.so' in the specified build directory.

    Args:
        library_name: Name of the plugin.
        c_header: C header signatures for the Python functions.
        python_wrapper: Python code wrapping the original function to be exposed.
        build_path: Path to the build directory.
        backend: Backend used by the generated C shared library.
    """
    try:
        header_file_path = write_c_header(build_path, library_name, c_header)
        compile_cffi_plugin(
            builder=configure_cffi_builder(c_header, library_name, header_file_path, rpath),
            python_wrapper=python_wrapper,
            build_path=str(build_path),
            library_name=library_name,
        )
    except Exception as e:
        logging.error(f"Error generating and compiling CFFI plugin: {e}")
        raise


def write_c_header(build_path: Path, library_name: str, c_header: str) -> Path:
    """Write the C header file to the specified path."""
    c_header_file = library_name + ".h"
    header_file_path = build_path / c_header_file
    with open(header_file_path, "w") as f:
        f.write(c_header)
    return header_file_path


def configure_cffi_builder(
    c_header: str, library_name: str, header_file_path: Path, rpath: str = ""
) -> cffi.FFI:
    """Configure and returns a CFFI FFI builder instance."""
    builder = cffi.FFI()
    extra_link_args = [f"-Wl,-rpath={rpath}"] if rpath else []
    builder.embedding_api(c_header)
    builder.set_source(
        library_name, f'#include "{header_file_path.name}"', extra_link_args=extra_link_args
    )
    return builder


def compile_cffi_plugin(
    builder: cffi.FFI, python_wrapper: str, build_path: str, library_name: str
) -> None:
    """Compile the CFFI plugin with the given configuration."""
    builder.embedding_init_code(python_wrapper)
    builder.compile(tmpdir=build_path, target=f"lib{library_name}.*", verbose=True)

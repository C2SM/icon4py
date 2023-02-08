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
import functools

import cffi


FFI_DEF_EXTERN_DECORATOR = "@ffi.def_extern()"

CFFI_GEN_DECORATOR = "@CffiMethod.register"


class CffiMethod:
    _registry = {}

    @classmethod
    def register(cls, func):
        try:
            cls._registry[func.__module__].append(func.__name__)
        except KeyError:
            cls._registry[func.__module__] = [func.__name__]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]


def with_cffi_gen(func):
    @functools.wraps(func)
    def _cffi_gen(*args, **kwargs):
        return func(*args, **kwargs)

    return _cffi_gen


def generate_and_compile_cffi_plugin(
    plugin_name: str, c_header: str, module_name: str, build_path="."
):
    """
    Create C shared library.

    Create a linkable C library and F90 interface for the functions in the python module
    {module_name} that are decorated with '@CffiMethod.register'.

    Args:
        plugin_name: name of the plugin, a linkable C library with the name
            'lib{plugin_name}.so' will be created in the {build_path} folder'
        c_header: C type header signature for the python functions.
        module_name:  python module name that contains python functions corresponding
            to the signature in the '{c_header}' string, these functions must be decorated
            with @CffiMethod.register and the file must contain the import
        build_path: *optional* path to build directory

    """
    python_src_file = f"{module_name.split('.')[-1]}.py"
    c_header_file = plugin_name + ".h"
    with open("/".join([build_path, c_header_file]), "w") as f:
        f.write(c_header)

    builder = cffi.FFI()

    builder.embedding_api(c_header)
    builder.set_source(plugin_name, f'#include "{c_header_file}"')

    with open(python_src_file) as f:
        module = f.read()

    module = f"from {plugin_name} import ffi\n{module}".replace(
        CFFI_GEN_DECORATOR, FFI_DEF_EXTERN_DECORATOR
    )

    builder.embedding_init_code(module)
    builder.compile(tmpdir=build_path, target=f"lib{plugin_name}.*", verbose=True)

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
import cffi


def cffi_plugin(plugin_name):
    from plugin_name import ffi

    def cffi_plugin_decorator(func):
        def plugin_wrapper(*args, **kwargs):
            return ffi.def_extern(func(*args, **kwargs))

        return plugin_wrapper


def compile_cffi_plugin(
    plugin_name: str, c_header: str, cffi_functions_file: str, build_path="."
):
    """
    Create C shared library.

    Create a linkable C library for the functions in {cffi_functions_file} that are decorated
    with '@ffi.def_extern' and correspond to a C signature in the header string

    Args:
        plugin_name: name of the plugin, a linkable C library with the name 'lib{plugin_name}.so' will be
            created in the build_path folder'
        c_header: C type header signature for the python functions.
        cffi_functions_file: input file that contains python functions correspondig to the signature in the '{c_header}'
            string, these functions must be decorated with @ffi.def_extern() and the file must contain the import
            'from {plugin_name} import ffi'
        build_path: *optional* path to build directory

    Returns:
    """
    c_header_file = plugin_name + ".h"
    with open("/".join([build_path, c_header_file]), "w") as f:
        f.write(c_header)

    builder = cffi.FFI()

    builder.embedding_api(c_header)
    builder.set_source(plugin_name, f'#include "{c_header_file}"')

    with open(cffi_functions_file) as f:
        module = f.read()

    import_str = f"from {plugin_name} import ffi\n"
    extern_decorator = "@ffi.def_extern()\n"
    module = f"{import_str}{module}"
    module.replace("def diffusion_init", extern_decorator + "def diffusion_init")


    builder.embedding_init_code(module)
    builder.compile(tmpdir=build_path, target=f"lib{plugin_name}.*", verbose=True)


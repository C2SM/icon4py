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

import pathlib

import click

from icon4pytools.py2fgen.cffi_utils import generate_and_compile_cffi_plugin
from icon4pytools.py2fgen.codegen import generate_and_write_f90_interface, generate_c_header
from icon4pytools.py2fgen.parsing import parse_function


@click.command(
    "py2fgen",
)
@click.argument("module_import_path", type=str)
@click.argument("function_name", type=str)
@click.argument(
    "build_path",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
    default=".",
)
def main(module_import_path: str, function_name: str, build_path: pathlib.Path,) -> None:
    """
    Generate C and F90 wrappers and C library for embedding the python MODULE in C and Fortran.

      Args:
          - module: name of the python module containing the methods to be embedded. Those
          methods have to be decorated with CffiMethod.register

          - build_path: directory where the generated code and compiled libraries are to be found.
    """
    build_path.mkdir(exist_ok=True, parents=True)
    plugin = parse_function(module_import_path, function_name)
    c_header = generate_c_header(plugin)
    generate_and_compile_cffi_plugin(plugin.name, c_header, module_import_path, str(build_path))
    generate_and_write_f90_interface(build_path, plugin)


if __name__ == "__main__":
    main()

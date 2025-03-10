# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import click

from icon4py.tools.common.utils import write_string
from icon4py.tools.py2fgen.generate import (
    generate_c_header,
    generate_f90_interface,
    generate_python_wrapper,
)
from icon4py.tools.py2fgen.parsing import get_cffi_description
from icon4py.tools.py2fgen.plugin import generate_and_compile_cffi_plugin


def parse_comma_separated_list(ctx: click.Context, param: click.Parameter, value: str) -> list[str]:
    # Splits the input string by commas and strips any leading/trailing whitespace from the strings
    return [item.strip() for item in value.split(",")]


@click.command("py2fgen")
@click.argument(
    "module_import_path",
    type=str,
)
@click.argument("functions", type=str, callback=parse_comma_separated_list)
@click.argument("plugin_name", type=str)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
    default=".",
    help="Specify the directory for generated code and compiled libraries.",
)
def main(
    module_import_path: str,
    functions: list[str],
    plugin_name: str,
    output_path: pathlib.Path,
) -> None:
    """Generate C and F90 wrappers and C library for embedding a Python module in C and Fortran."""
    output_path.mkdir(exist_ok=True, parents=True)

    plugin = get_cffi_description(module_import_path, functions, plugin_name)

    c_header = generate_c_header(plugin)
    python_wrapper = generate_python_wrapper(plugin)
    f90_interface = generate_f90_interface(plugin)

    generate_and_compile_cffi_plugin(plugin.plugin_name, c_header, python_wrapper, output_path)
    write_string(f90_interface, output_path, f"{plugin.plugin_name}.f90")
    write_string(python_wrapper, output_path, f"{plugin.plugin_name}.py")


if __name__ == "__main__":
    main()

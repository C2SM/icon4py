# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import click

from icon4pytools.common.utils import write_string
from icon4pytools.py2fgen.generate import (
    generate_c_header,
    generate_f90_interface,
    generate_python_wrapper,
)
from icon4pytools.py2fgen.parsing import parse
from icon4pytools.py2fgen.plugin import generate_and_compile_cffi_plugin
from icon4pytools.py2fgen.wrappers import common


def parse_comma_separated_list(ctx, param, value) -> list[str]:
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
@click.option(
    "--backend",
    "-b",
    type=click.Choice([e.name for e in common.GT4PyBackend], case_sensitive=False),
    default="CPU",
    help="Set the backend to use, thereby unpacking Fortran pointers into NumPy or CuPy arrays respectively.",
)
@click.option(
    "--debug-mode",
    "-d",
    is_flag=True,
    help="Enable debug mode to log additional Python runtime information.",
)
@click.option(
    "--profile",
    "-p",
    is_flag=True,
    help="Profile granule runtime and unpacking Fortran pointers into NumPy or CuPy arrays.",
)
@click.option("--limited-area", is_flag=True, help="Enable limited area mode.")
def main(
    module_import_path: str,
    functions: list[str],
    plugin_name: str,
    output_path: pathlib.Path,
    debug_mode: bool,
    backend: str,
    limited_area: str,
    profile: bool,
) -> None:
    """Generate C and F90 wrappers and C library for embedding a Python module in C and Fortran."""
    output_path.mkdir(exist_ok=True, parents=True)

    plugin = parse(module_import_path, functions, plugin_name)

    c_header = generate_c_header(plugin)
    python_wrapper = generate_python_wrapper(plugin, backend, debug_mode, limited_area, profile)
    f90_interface = generate_f90_interface(plugin, limited_area)

    generate_and_compile_cffi_plugin(
        plugin.plugin_name, c_header, python_wrapper, output_path, backend
    )
    write_string(f90_interface, output_path, f"{plugin.plugin_name}.f90")
    write_string(python_wrapper, output_path, f"{plugin.plugin_name}.py")


if __name__ == "__main__":
    main()

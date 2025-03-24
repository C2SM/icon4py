# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import click

from icon4py.tools.py2fgen import _codegen, _generator, _utils


logger = _utils.setup_logger("py2fgen")


@click.command("py2fgen")
@click.argument(
    "module_import_path",
    type=str,
)
@click.argument("functions", type=str, callback=_utils.parse_comma_separated_list)
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

    plugin = _generator.get_cffi_description(module_import_path, functions, plugin_name)

    logger.info("Generating C header...")
    c_header = _codegen.generate_c_header(plugin)
    logger.info("Generating Python wrapper...")
    python_wrapper = _codegen.generate_python_wrapper(plugin)
    _utils.write_file(python_wrapper, output_path, f"{plugin.library_name}.py")
    logger.info("Generating Fortran interface...")
    f90_interface = _codegen.generate_f90_interface(plugin)
    _utils.write_file(f90_interface, output_path, f"{plugin.library_name}.f90")

    logger.info("Compiling CFFI dynamic library...")
    _generator.generate_and_compile_cffi_plugin(
        plugin.library_name, c_header, python_wrapper, output_path
    )


if __name__ == "__main__":
    main()

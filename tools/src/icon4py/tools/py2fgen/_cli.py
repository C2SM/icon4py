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
@click.argument("library_name", type=str)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
    default=".",
    help="Specify the directory for generated code and compiled libraries.",
)
@click.option(
    "--rpath",
    "-r",
    type=str,
    default="",
    help="Specify an rpath for the compiled library. If not set, no rpath is added.",
)
@click.option(
    "--regenerate",
    is_flag=True,
    default=False,
    help="Force regeneration of all files and recompilation, even if they are up to date.",
)
def main(
    module_import_path: str,
    functions: list[str],
    library_name: str,
    output_path: pathlib.Path,
    rpath: str,
    regenerate: bool,
) -> None:
    """Generate C and F90 wrappers and C library for embedding a Python module in C and Fortran."""
    output_path.mkdir(exist_ok=True, parents=True)
    plugin = _generator.get_cffi_description(module_import_path, functions, library_name)

    logger.info("Generating C header...")
    c_header = _codegen.generate_c_header(plugin)
    logger.info("Generating Python wrapper...")
    python_wrapper = _codegen.generate_python_wrapper(plugin)
    logger.info("Generating Fortran interface...")
    f90_interface = _codegen.generate_f90_interface(plugin)

    if regenerate:
        logger.info("Force regeneration requested.")

    any_changed = False
    for content, fname, label in [
        (python_wrapper, f"{plugin.library_name}.py", "Python wrapper"),
        (f90_interface, f"{plugin.library_name}.f90", "Fortran interface"),
    ]:
        changed = _utils.write_file_if_changed(content, output_path, fname, force=regenerate)
        logger.info("%s %s.", label, "changed" if changed else "is up to date")
        any_changed |= changed

    compilation_outputs_exist = (
        (output_path / f"{plugin.library_name}.h").exists()
        # Glob matches the shared library (.so on Linux, .dylib on macOS).
        # Note: may also match other files like .c or .o if they
        # happen to be prefixed with 'lib{plugin.library_name}'.
        and any(output_path.glob(f"lib{plugin.library_name}.*"))
    )
    if not compilation_outputs_exist:
        logger.info("Compilation outputs missing.")

    if any_changed or not compilation_outputs_exist:
        logger.info("Compiling CFFI dynamic library...")
        _generator.generate_and_compile_cffi_plugin(
            plugin.library_name, c_header, python_wrapper, output_path, rpath
        )
    else:
        logger.info("All generated files are up to date. Skipping compilation.")


if __name__ == "__main__":
    main()

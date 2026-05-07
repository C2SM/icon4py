# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
import sysconfig

import click

from icon4py.tools.py2fgen import _codegen, _generator, _outputs, _utils
from icon4py.tools.py2fgen._outputs import ArtifactKind


logger = _utils.setup_logger("py2fgen", log_level=logging.INFO)


_FILE_PATH = click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path)


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
@click.option(
    "--compile",
    "compile_",
    is_flag=True,
    default=False,
    help=(
        "Also compile the shared library (lib<library_name>.so). "
        "Cannot be combined with any --output-<kind>."
    ),
)
@click.option(
    "--output-py",
    type=_FILE_PATH,
    default=None,
    help="Override path for the Python wrapper (.py). Selective mode (no compile).",
)
@click.option(
    "--output-f90",
    type=_FILE_PATH,
    default=None,
    help="Override path for the Fortran interface (.f90). Selective mode (no compile).",
)
@click.option(
    "--output-c",
    type=_FILE_PATH,
    default=None,
    help="Override path for the CFFI C source (.c). Selective mode (no compile).",
)
@click.option(
    "--output-h",
    type=_FILE_PATH,
    default=None,
    help="Override path for the C header (.h). Selective mode (no compile).",
)
def main(
    module_import_path: str,
    functions: list[str],
    library_name: str,
    output_path: pathlib.Path,
    rpath: str,
    regenerate: bool,
    compile_: bool,
    output_py: pathlib.Path | None,
    output_f90: pathlib.Path | None,
    output_c: pathlib.Path | None,
    output_h: pathlib.Path | None,
) -> None:
    """Generate C and F90 wrappers and C library for embedding a Python module in C and Fortran."""
    output_path.mkdir(exist_ok=True, parents=True)

    overrides = {
        ArtifactKind.PY: output_py,
        ArtifactKind.F90: output_f90,
        ArtifactKind.C: output_c,
    }
    if compile_ and (any(p is not None for p in overrides.values()) or output_h is not None):
        raise click.UsageError(
            "--compile cannot be combined with --output-<kind>; "
            "selective emission produces sources only."
        )

    plan = _outputs.resolve(
        library_name=library_name,
        output_path=output_path,
        overrides=overrides,
        output_h=output_h,
        compile_lib=compile_,
    )

    plugin = _generator.get_cffi_description(module_import_path, functions, library_name)

    if regenerate:
        logger.info("Force regeneration requested.")

    any_changed = False

    if ArtifactKind.PY in plan.paths:
        logger.info("Generating Python wrapper...")
        python_wrapper = _codegen.generate_python_wrapper(plugin)
        changed = _utils.write_path_if_changed(
            python_wrapper, plan.paths[ArtifactKind.PY], force=regenerate
        )
        logger.info("Python wrapper %s.", "changed" if changed else "is up to date")
        any_changed |= changed
    else:
        # Compilation needs the Python wrapper content even if we don't write it.
        python_wrapper = _codegen.generate_python_wrapper(plugin) if plan.compile else ""

    if ArtifactKind.F90 in plan.paths:
        logger.info("Generating Fortran interface...")
        f90_interface = _codegen.generate_f90_interface(plugin)
        changed = _utils.write_path_if_changed(
            f90_interface, plan.paths[ArtifactKind.F90], force=regenerate
        )
        logger.info("Fortran interface %s.", "changed" if changed else "is up to date")
        any_changed |= changed

    needs_c = ArtifactKind.C in plan.paths
    if needs_c or plan.compile:
        c_header = _codegen.generate_c_header(plugin)
        _emit_c_artifacts(
            plan=plan,
            library_name=library_name,
            output_path=output_path,
            c_header=c_header,
            python_wrapper=python_wrapper,
            rpath=rpath,
            any_changed=any_changed,
        )


def _emit_c_artifacts(
    *,
    plan: _outputs.OutputPlan,
    library_name: str,
    output_path: pathlib.Path,
    c_header: str,
    python_wrapper: str,
    rpath: str,
    any_changed: bool,
) -> None:
    h_path = plan.h_path
    assert h_path is not None  # set whenever C is in plan.paths or compile is True

    if plan.compile:
        shared_lib_path = (
            output_path / f"lib{library_name}{sysconfig.get_config_var('SHLIB_SUFFIX')}"
        )
        compilation_outputs_exist = h_path.exists() and shared_lib_path.exists()
        if not compilation_outputs_exist:
            logger.info("Compilation outputs missing.")
        if any_changed or not compilation_outputs_exist:
            logger.info("Compiling CFFI dynamic library...")
            _generator.generate_and_compile_cffi_plugin(
                library_name, c_header, python_wrapper, output_path, h_path, rpath
            )
        else:
            logger.info("All generated files are up to date. Skipping compilation.")
        return

    c_path = plan.paths[ArtifactKind.C]
    if any_changed or not c_path.exists() or not h_path.exists():
        logger.info("Generating C source and header files...")
        _generator.generate_cffi_source(
            library_name, c_header, python_wrapper, c_path, h_path, rpath
        )
    else:
        logger.info("All generated files are up to date. Skipping C code generation.")


if __name__ == "__main__":
    main()

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""
Collection of all Fortran bindings for ICON4Py.

TODO(havogt): refactor py2fgen to generate a single module from functions in different files.

Runnable as ``python -m icon4py.bindings.all_bindings`` to emit the .f90 and/or
.c source for the bundled bindings at user-chosen paths.
"""

from __future__ import annotations

import pathlib

import click

from icon4py.bindings.diffusion_wrapper import diffusion_init, diffusion_run
from icon4py.bindings.dycore_wrapper import solve_nh_init, solve_nh_run
from icon4py.bindings.grid_wrapper import grid_init
from icon4py.tools.py2fgen._cli import main as _py2fgen_main


__all__ = [
    "diffusion_init",
    "diffusion_run",
    "grid_init",
    "solve_nh_init",
    "solve_nh_run",
]


LIBRARY_NAME = "icon4py_bindings"


_FILE_PATH = click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path)


@click.command(name="all_bindings")
@click.option(
    "--output-f90",
    type=_FILE_PATH,
    default=None,
    help="Path for the generated Fortran interface (.f90).",
)
@click.option(
    "--output-c",
    type=_FILE_PATH,
    default=None,
    help="Path for the generated CFFI C source (.c). The .h header is written alongside.",
)
def main(output_f90: pathlib.Path | None, output_c: pathlib.Path | None) -> None:
    """Generate the icon4py Fortran/C binding sources at user-chosen paths."""
    if output_f90 is None and output_c is None:
        raise click.UsageError("specify --output-f90 and/or --output-c")

    _py2fgen_main.callback(  # type: ignore[misc]
        module_import_path="icon4py.bindings.all_bindings",
        functions=list(__all__),
        library_name=LIBRARY_NAME,
        output_path=pathlib.Path(),
        rpath="",
        regenerate=False,
        compile_=False,
        output_py=None,
        output_f90=output_f90,
        output_c=output_c,
        output_h=None,
    )


if __name__ == "__main__":
    main()

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""CLI entrypoint that generates the combined icon4py Fortran/C bindings.

Run as ``python -m icon4py.bindings.all_bindings --output-f90 X --output-c Y``.
"""

from __future__ import annotations

import pathlib

import click

from icon4py.bindings import diffusion_wrapper, dycore_wrapper, grid_wrapper
from icon4py.tools import py2fgen


FUNCTIONS = [
    diffusion_wrapper.diffusion_init,
    diffusion_wrapper.diffusion_run,
    grid_wrapper.grid_init,
    dycore_wrapper.solve_nh_init,
    dycore_wrapper.solve_nh_run,
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

    plugin = py2fgen.get_cffi_description(FUNCTIONS, LIBRARY_NAME)
    h_basename = output_c.with_suffix(".h").name if output_c is not None else f"{LIBRARY_NAME}.h"
    sources = py2fgen.render(plugin, h_basename=h_basename)

    # write_if_changed avoids touching mtimes (and triggering downstream
    # rebuilds) when the rendered content matches what is already on disk.
    if output_f90 is not None:
        py2fgen.write_if_changed(sources.f90, output_f90)
    if output_c is not None:
        py2fgen.write_if_changed(sources.c, output_c)
        py2fgen.write_if_changed(sources.h, output_c.with_suffix(".h"))


if __name__ == "__main__":
    main()

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

from icon4py.common.logger import setup_logger
from icon4py.liskov.pipeline.collection import (
    load_gt4py_stencils,
    parse_fortran_file,
    run_code_generation,
)


logger = setup_logger(__name__)


@click.command("icon_liskov")
@click.argument(
    "input_path",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.argument(
    "mode",
    type=str,
)
@click.option(
    "--profile",
    "-p",
    is_flag=True,
    help="Add nvtx profile statements to integration code.",
)
@click.option(
    "--metadatagen",
    "-m",
    is_flag=True,
    help="Add metadata header with information about program (requires git).",
)
def main(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    mode: str,
    profile: bool,
    metadatagen: bool,
) -> None:
    """Command line interface for interacting with the ICON-Liskov DSL Preprocessor.

    Usage:
        icon_liskov <input> <output> <mode> [-p] [-m]

    Options:
        -p --profile Add nvtx profile statements to stencils.
        -m --metadatagen Add metadata header with information about program (requires git).

    Arguments:
        input_path: Path to the input file to process.
        output_path: Path to the output file to generate.
        mode: Code generation mode to use. Available options are serialisation or integration.
    """
    if mode not in ["serialisation", "integration"]:
        raise ValueError("Invalid mode: must be 'serialisation' or 'integration'")

    def run_serialisation():
        iface = parse_fortran_file(input_path, output_path, mode)
        run_code_generation(input_path, output_path, mode, iface)

    def run_integration():
        iface = parse_fortran_file(input_path, output_path, mode)
        iface_gt4py = load_gt4py_stencils(iface)
        run_code_generation(
            input_path,
            output_path,
            mode,
            iface_gt4py,
            profile=profile,
            metadatagen=metadatagen,
        )

    mode_dispatcher = {
        "serialisation": run_serialisation,
        "integration": run_integration,
    }

    mode_dispatcher[mode]()


if __name__ == "__main__":
    main()

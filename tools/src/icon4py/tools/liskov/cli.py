# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Optional

import click

from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.external.exceptions import MissingCommandError
from icon4pytools.liskov.pipeline.collection import (
    parse_fortran_file,
    process_stencils,
    run_code_generation,
)


logger = setup_logger(__name__)


def split_comma(ctx, param, value) -> Optional[tuple[str]]:
    return tuple(v.strip() for v in value.split(",")) if value else None


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Command line interface for interacting with the ICON-Liskov DSL Preprocessor."""
    if ctx.invoked_subcommand is None:
        raise MissingCommandError(
            "Need to choose one of the following commands:\nintegrate\nserialise"
        )


@main.command()
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
    help="Add metadata header with information about program.",
)
@click.option(
    "--fused/--unfused",
    "-f/-u",
    default=False,
    help="Adds fused or unfused stencils.",
)
@click.option(
    "--optional-modules-to-enable",
    callback=split_comma,
    help="Specify a list of comma-separated optional DSL modules to enable.",
)
@click.option(
    "--verification/--substitution",
    "-v/-s",
    default=False,
    help="Adds verification runs and checks.",
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
def integrate(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    fused: bool,
    verification: bool,
    profile: bool,
    metadatagen: bool,
    optional_modules_to_enable: Optional[tuple[str]],
) -> None:
    mode = "integration"
    iface = parse_fortran_file(input_path, output_path, mode)
    iface_gt4py = process_stencils(
        iface, fused, optional_modules_to_enable=optional_modules_to_enable
    )
    run_code_generation(
        input_path,
        output_path,
        mode,
        iface_gt4py,
        profile=profile,
        verification=verification,
        metadatagen=metadatagen,
    )


@main.command()
@click.option(
    "--multinode",
    is_flag=True,
    type=bool,
    help="Specify whether it is a multinode run. Will generate mpi rank information.",
    default=False,
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
def serialise(input_path: pathlib.Path, output_path: pathlib.Path, multinode: bool) -> None:
    mode = "serialisation"
    iface = parse_fortran_file(input_path, output_path, mode)
    run_code_generation(input_path, output_path, mode, iface, multinode=multinode)


if __name__ == "__main__":
    main()

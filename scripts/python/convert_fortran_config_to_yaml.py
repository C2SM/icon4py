#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert serialized Fortran namelist dumps into a single YAML driver config.

Reads the ``NAMELIST_ICON_output_atm.json``, ``icon_master.namelist.json`` and
``NAMELIST_expname.json`` files (see ``icon4py.model.common.utils.fortran_config``)
from a namelist directory -- typically an experiment's serialized data directory,
see ``icon4py.model.driver.config.read_experiment_config_from_fortran`` -- and
writes the equivalent ``icon4py.model.driver.config.ExperimentConfig`` as YAML, for
use with ``icon4py-driver --config-file-path``.

    ./scripts/run convert-fortran-config-to-yaml <namelist-dir> -o config.yml
"""

from __future__ import annotations

import pathlib
import sys
from typing import Annotated

import typer


cli = typer.Typer(no_args_is_help=True, help=__doc__)


@cli.command()
def convert_fortran_config_to_yaml(
    namelist_dir: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Directory with the serialized Fortran namelist JSON dumps.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option("-o", "--output", help="Path the YAML config is written to."),
    ] = pathlib.Path("config.yml"),
    enable_profiling: Annotated[
        bool,
        typer.Option(help="Include the profiling options section in the generated config."),
    ] = False,
    enable_statistics_output: Annotated[
        bool,
        typer.Option(help="Enable variable-statistics logging in the generated config."),
    ] = False,
) -> None:
    """Convert the Fortran namelist dumps in NAMELIST_DIR into a YAML config at OUTPUT."""
    # Import here to reduce startup time for the CLI.
    from icon4py.model.common.config import config_io  # noqa: PLC0415 [import-outside-top-level]
    from icon4py.model.driver import config as driver_config  # noqa: PLC0415

    config = driver_config.read_experiment_config_from_fortran(
        namelist_dir,
        enable_profiling=enable_profiling,
        enable_statistics_output=enable_statistics_output,
    )
    output.write_text(config_io.write_yaml_str(config))
    typer.echo(f"Wrote YAML config for '{namelist_dir}' to '{output}'.")


if __name__ == "__main__":
    sys.exit(cli())

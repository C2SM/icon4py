#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that all icon4py package and dependency versions are consistent."""

from __future__ import annotations

import sys

import typer
from bump_versions import _find_package_dirs, _validate_package_versions


cli = typer.Typer(no_args_is_help=True, help=__doc__)


@cli.command()
def check_versions() -> None:
    """Verify all package versions and pinned icon4py dependencies agree."""
    current_version = _validate_package_versions(_find_package_dirs())
    typer.echo(f"All icon4py package versions are consistent at {current_version}.")


if __name__ == "__main__":
    sys.exit(cli())

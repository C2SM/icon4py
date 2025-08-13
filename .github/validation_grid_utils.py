#! /usr/bin/env -S uv run -q --script

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: ERA001

# /// script
# requires-python = ">=3.10"
# dependencies = [
# "typer>=0.10",
# "icon4py-testing",
# ]
# [tool.uv.sources]
# icon4py-testing = {path = "../model/testing"}
# ///


import hashlib
import os
import pathlib
import sys

import typer

from icon4py.model.testing import config, definitions, grid_utils


VALIDATION_GRIDS = (
    definitions.Grids.R02B04_GLOBAL,
    definitions.Grids.MCH_CH_R04B09_DSL,
    definitions.Grids.MCH_OPR_R04B07_DOMAIN01,
)  # change to MCH_OPR_R04B07_DOMAIN01
app = typer.Typer()


@app.command(name="cache-key")
def cache_key() -> None:
    """Generate a cache key for the Github action cache based on grid file name and download URI."""
    d = "_".join(grid.name + grid.uri for grid in VALIDATION_GRIDS)
    hexdigest = hashlib.md5(d.encode()).hexdigest()
    print(hexdigest)


@app.command(name="download")
def download_validation_grids() -> None:
    """Effectively download the validation grid files."""
    config.TEST_DATA_PATH = pathlib.Path(os.getcwd()) / definitions.DEFAULT_TEST_DATA_FOLDER
    for grid in VALIDATION_GRIDS:
        print(f"downloading and unpacking {grid.name}")
        fname = grid_utils._download_grid_file(grid.name)
        print(f"done - downloaded {fname}")


if __name__ == "__main__":
    sys.exit(app())

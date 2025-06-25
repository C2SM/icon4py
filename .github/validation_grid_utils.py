#! /usr/bin/env -S uv run -q --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
# "typer>=0.10",
# "icon4py-testing",
# ]
# [tool.uv.sources]
# icon4py-testing = {path = "../model/testing"}
# ///


import typer
import sys

from icon4py.model.testing import grid_utils
from icon4py.model.testing import definitions
import hashlib

VALIDATION_GRIDS = (definitions.Grids.R02B04_GLOBAL, definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.MCH_OPR_R04B07_DOMAIN01) # change to MCH_OPR_R04B07_DOMAIN01
app = typer.Typer()
@app.command(name="cache-key")
def cache_key()->None:
    """ Generate a cache key for the Github action cache based on grid file name and download URI."""
    d = "_".join(grid.name + grid.uri for grid in VALIDATION_GRIDS)
    hexdigest = hashlib.md5(d.encode()).hexdigest()
    print(hexdigest)


@app.command(name="download")
def download_validation_grids()->None:
    """Effectively download the validation grid files."""
    for grid in VALIDATION_GRIDS:
        print(f"downloading and unpacking {grid.name}")
        fname = grid_utils._download_grid_file(grid.name)
        print(f"done - downloaded {fname}")




if __name__=="__main__":
    sys.exit(app())

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
import enum
import sys

from icon4py.model.testing import grid_utils
from icon4py.model.testing import definitions
import hashlib

VALIDATION_GRIDS = (definitions.Grids.R02B04_GLOBAL, definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.MCH_OPR_R04B07_DOMAIN01) # change to MCH_OPR_R04B07_DOMAIN01
app = typer.Typer()

def cache_key()->None:
    d = "_".join(grid.name + grid.uri for grid in VALIDATION_GRIDS)
    hexdigest = hashlib.md5(d.encode()).hexdigest()
    print(hexdigest)



def download_validation_grids()->None:
    for grid in VALIDATION_GRIDS:
        print(f"downloading and unpacking {grid.name}")
        fname = grid_utils._download_grid_file(grid.name)
        print(f"done - downloaded {fname}")

class Mode(enum.Enum):
    DOWNLOAD = "download"
    CACHE_KEY = "cache-key"

@app.command()
def main(option: Mode = Mode.DOWNLOAD) -> int:
    """
    Functionality for grid file download.

    The script has to options:

    - `cache-key`: generate the cache key for the github cache action. The cache key is based on the download URI and the grid name

    - `download` : effectively download the files
    """
    try:
        match(option):
            case Mode.DOWNLOAD:
                download_validation_grids()
            case Mode.CACHE_KEY:
                cache_key()
            case _:
                raise ValueError(f"Unknown option: {option}")
        return 0
    except ValueError as e:
        message, exit_code = e.args
        print(f"[ERROR] {message}")
        return exit_code



if __name__=="__main__":
    sys.exit(app())

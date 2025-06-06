import enum

import click


from icon4py.model.testing import grid_utils
from icon4py.model.testing import definitions
import hashlib

validation_grids = (definitions.Grids.R02B04_GLOBAL, definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.MCH_OPR_R04B07_DOMAIN01) # change to MCH_OPR_R04B07_DOMAIN01

def cache_key()->str:
    d = "_".join(grid.name + grid.uri for grid in validation_grids)
    hexdigest = hashlib.md5(d.encode()).hexdigest()
    print(hexdigest)
    return hexdigest


def download_validation_grids()->str:
    for grid in validation_grids:
        print(f"downloading and unpacking {grid.name}")
        fname = grid_utils._download_grid_file(grid.name)
        print(f"done - downloaded {fname}")
    return "OK"

OPTIONS = ("download", "cache-key")


@click.command()
@click.option("--option", type=click.Choice(OPTIONS, case_sensitive=False),
              default=OPTIONS[0], help=f"Choose the mode of operation: {OPTIONS}.")
def main(option: str) -> str:

    match(option):
        case "download":
            return download_validation_grids()
        case "cache-key":
            return cache_key()
        case _:
            raise ValueError(f"Unknown option: {option}")



if __name__=="__main__":
    main()

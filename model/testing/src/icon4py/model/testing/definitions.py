# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import os
import pathlib
import types
import uuid
from collections.abc import Mapping
from typing import Final, Literal


_TEST_UTILS_PATH: Final = pathlib.Path(__file__) / ".."
_MODEL_PATH: Final = _TEST_UTILS_PATH / ".."
_COMMON_PATH: Final = _MODEL_PATH / ".." / ".." / ".." / ".."

DEFAULT_TEST_DATA_FOLDER: Final = "testdata"
TEST_DATA_PATH: Final[pathlib.Path] = pathlib.Path(
    os.getenv("TEST_DATA_PATH") or (_COMMON_PATH / ".." / DEFAULT_TEST_DATA_FOLDER)
)
GRIDS_PATH: Final[pathlib.Path] = TEST_DATA_PATH / "grids"
SERIALIZED_DATA_PATH: Final[pathlib.Path] = TEST_DATA_PATH / "ser_icondata"


class GridKind(enum.Enum):
    REGIONAL = "REGIONAL"
    GLOBAL = "GLOBAL"


@dataclasses.dataclass
class Grid:
    name: str
    description: str
    kind: GridKind
    sizes: Mapping[Literal["cell", "edge", "vertex"], int]
    file_name: str | None = None


class Grids:
    SIMPLE: Final = Grid(name="<simple-grid>", description="", sizes={})
    ICON: Final = Grid(name="icon_grid", description="", sizes={})
    ICON_GLOBAL: Final = Grid(name="icon_grid_global", description="", sizes={})
    REGIONAL: Final = Grid(name="regional_grid", description="", sizes={})
    GLOBAL: Final = Grid(name="global_grid", description="", sizes={})
    MC_CH_R04B09_DSL: Final = Grid(name="MC_CH_R04B09_DSL_GRID_URI", description="", sizes={})
    R02B04_GLOBAL: Final = Grid(name="R02B04_GLOBAL_GRID_URI", description="", sizes={})
    TORUS_100X116_1000M: Final = Grid(name="TORUS_100X116_1000M_GRID_URI", description="", sizes={})
    TORUS_50000x5000: Final = Grid(name="TORUS_50000x5000_RES500", description="", sizes={})


@dataclasses.dataclass
class Experiment:
    name: str
    description: str
    grid: Grid
    num_levels: int
    partitioned_data: Mapping[int, str] | None = None


class Experiments:
    GLOBAL: Final = Experiment(name="exclaim_ape_R02B04", description="", grid=None)
    REGIONAL: Final = Experiment(
        name="mch_ch_r04b09_dsl", description="", grid=Grids.MC_CH_R04B09_DSL
    )
    R02B04: Final = Experiment(name="r02b04_global", description="", grid=Grids.R02B04_GLOBAL)
    JABW: Final = Experiment(name="jabw_R02B04", description="", grid=None)
    GAUSS3D: Final = Experiment(name="gauss3d_torus", description="", grid=None)
    WEISMAN_KLEMP: Final = Experiment(
        name="weisman_klemp_torus", description="", grid=Grids.TORUS_50000x5000
    )


# GRID_DATA_URIS: Final[Mapping[GridName, Mapping[int, str]]] = {
#     GridName.MC_CH_R04B09_DSL: {1: "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"},
#     GridName.R02B04_GLOBAL: {1: "https://polybox.ethz.ch/index.php/s/AKAO6ImQdIatnkB/download"},
#     GridName.TORUS_100X116_1000M: {1: "https://polybox.ethz.ch/index.php/s/yqvotFss9i1OKzs/download"},
#     GridName.TORUS_50000x5000: {1: "https://polybox.ethz.ch/index.php/s/eclzK00TM9nnLtE/download"},
#     GridName.ICON: {
#         1: "https://polybox.ethz.ch/index.php/s/f42nsmvgOoWZPzi/download",
#         2: "https://polybox.ethz.ch/index.php/s/P6F6ZbzWHI881dZ/download",
#         4: "https://polybox.ethz.ch/index.php/s/NfES3j9no15A0aX/download",
#     },
#     GridName.GLOBAL: {1: "https://polybox.ethz.ch/index.php/s/2n2WpTgZFlTCTHu/download"},
#     GridName.DATA_URIS_JABW: {1: "https://polybox.ethz.ch/index.php/s/5W3Z2K6pyo0egzo/download"},
#     GridName.DATA_URIS_GAUSS3D: {1: "https://polybox.ethz.ch/index.php/s/ZuqDIREPVits9r0/download"},
#     GridName.DATA_URIS_WK: {1: "https://polybox.ethz.ch/index.php/s/ByLnyii7MMRHJbK/download"},
# }
# GRID_IDS = {
#     ExperimentName.GLOBAL: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
#     ExperimentName.REGIONAL: uuid.UUID("f2e06839-694a-cca1-a3d5-028e0ff326e0"),
#     ExperimentName.JABW: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
#     ExperimentName.GAUSS3D: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
#     ExperimentName.WEISMAN_KLEMP: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
# }

# MCH_CH_R04B09_LEVELS = 65
# GLOBAL_NUM_LEVELS = 60
# GLOBAL_GRIDFILE = "icon_grid_0013_R02B04_R.nc"
# REGIONAL_GRIDFILE = "grid.nc"

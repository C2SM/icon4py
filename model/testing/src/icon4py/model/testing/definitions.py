# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
from collections.abc import Mapping
from typing import Final


class Experiment(enum.Enum):
    """Enumeration of available experiments."""
    GLOBAL = "exclaim_ape_R02B04"
    REGIONAL = "mch_ch_r04b09_dsl"
    R02B04 = "r02b04_global"
    JABW = "jabw_R02B04"
    GAUSS3D = "gauss3d_torus"
    WEISMAN_KLEMP = "weisman_klemp_torus"



Experiment.GLOBAL = "exclaim_ape_R02B04"
Experiment.REGIONAL = "mch_ch_r04b09_dsl"
Experiment.R02B04 = "r02b04_global"
Experiment.JABW = "jabw_R02B04"
Experiment.GAUSS3D = "gauss3d_torus"
Experiment.WEISMAN_KLEMP = "weisman_klemp_torus"

MC_CH_R04B09_DSL_GRID_URI = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
R02B04_GLOBAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/AKAO6ImQdIatnkB/download"
TORUS_100X116_1000M_GRID_URI = "https://polybox.ethz.ch/index.php/s/yqvotFss9i1OKzs/download"
TORUS_50000x5000_RES500 = "https://polybox.ethz.ch/index.php/s/eclzK00TM9nnLtE/download"
GRID_URIS = {
    Experiment.REGIONAL: MC_CH_R04B09_DSL_GRID_URI,
    Experiment.R02B04: R02B04_GLOBAL_GRID_URI,
    Experiment.WEISMAN_KLEMP: TORUS_50000x5000_RES500,  # TODO: check
}


#: Mapping of ... to their data URIs
DATA_URIS: Final[Mapping[int, str]] = {
    1: "https://polybox.ethz.ch/index.php/s/f42nsmvgOoWZPzi/download",
    2: "https://polybox.ethz.ch/index.php/s/P6F6ZbzWHI881dZ/download",
    4: "https://polybox.ethz.ch/index.php/s/NfES3j9no15A0aX/download",
}
DATA_URIS_APE: Final[Mapping[int, str]] = {
    1: "https://polybox.ethz.ch/index.php/s/2n2WpTgZFlTCTHu/download"
}
DATA_URIS_JABW: Final[Mapping[int, str]] = {
    1: "https://polybox.ethz.ch/index.php/s/5W3Z2K6pyo0egzo/download"
}
DATA_URIS_GAUSS3D: Final[Mapping[int, str]] = {
    1: "https://polybox.ethz.ch/index.php/s/ZuqDIREPVits9r0/download"
}
DATA_URIS_WK: Final[Mapping[int, str]] = {
    1: "https://polybox.ethz.ch/index.php/s/ByLnyii7MMRHJbK/download"
}

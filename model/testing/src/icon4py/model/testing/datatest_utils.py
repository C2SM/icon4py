# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
import re
import uuid
from typing import TYPE_CHECKING, Optional

from gt4py.next import backend as gtx_backend

from icon4py.model.common.decomposition import definitions as decomposition


if TYPE_CHECKING:
    from icon4py.model.testing import serialbox

DEFAULT_TEST_DATA_FOLDER = "testdata"
GLOBAL_EXPERIMENT = "exclaim_ape_R02B04"
REGIONAL_EXPERIMENT = "mch_ch_r04b09_dsl"
R02B04_GLOBAL = "r02b04_global"
R02B07_GLOBAL = "r02b07_global"
ICON_CH2_SMALL = "mch_opr_r4b7"
REGIONAL_BENCHMARK = "opr_r19b08"
JABW_EXPERIMENT = "jabw_R02B04"
GAUSS3D_EXPERIMENT = "gauss3d_torus"
WEISMAN_KLEMP_EXPERIMENT = "weisman_klemp_torus"

TORUS_100X116_1000M_GRID_URI = "https://polybox.ethz.ch/index.php/s/yqvotFss9i1OKzs/download"
TORUS_50000x5000_RES500 = "https://polybox.ethz.ch/index.php/s/eclzK00TM9nnLtE/download"


# GRID URIs for global grids
#: small global grid use in exclaim_ape, JABW test case: num_cells = 20480
#  the origin of this file is unclear (source = icon-dev)
R02B04_GLOBAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/BRiF7XrCCpGqpEF/download"

#: large grid that should fit a single GPU for a simple atmospheric test case (Jablonowski-Williamson): num_cells = 1310720
#: this is generated by the grid_generator of MPI-M
R02B07_GLOBAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/RMqNbaeHLD5tDd6/download"

# GRID URIS for regional grids (MCH)
# those files are generated by (different versions of) icontools, (DWD grid file generator), see source attribute in the file.

#: used in mch_icon-ch2 experiment (2km resolution MCH production runs): num_cells = 283876

R19_B07_MCH_LOCAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/tFQian4aDzTES6c/download"

#: used in mch_icon-ch2-small experiment (for valdiation): num_cells = 10700
MCH_OPR_R04B07_DOMAIN01_GRID_URI = "https://polybox.ethz.ch/index.php/s/ZL7LeEDijGCSJGz/download"

#: used in opr_r19b08 ("old" experiment) that should fit and saturate a single GPU run: num_cells = 44528
DOMAIN01_GRID_URI = "https://polybox.ethz.ch/index.php/s/P6XfWcYjnrsNmeX/download"

#: used in legacy mch_ch_rO4b09_dsl experiment: (num_cells = 20896)
MC_CH_R04B09_DSL_GRID_URI = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"


GRID_URIS = {
    REGIONAL_EXPERIMENT: MC_CH_R04B09_DSL_GRID_URI,
    R02B04_GLOBAL: R02B04_GLOBAL_GRID_URI,
    R02B07_GLOBAL: R02B07_GLOBAL_GRID_URI,
    ICON_CH2_SMALL: MCH_OPR_R04B07_DOMAIN01_GRID_URI,
    REGIONAL_BENCHMARK: DOMAIN01_GRID_URI,
    WEISMAN_KLEMP_EXPERIMENT: TORUS_50000x5000_RES500,  # TODO: check
}

GRID_IDS = {
    GLOBAL_EXPERIMENT: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    REGIONAL_EXPERIMENT: uuid.UUID("f2e06839-694a-cca1-a3d5-028e0ff326e0"),
    JABW_EXPERIMENT: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    GAUSS3D_EXPERIMENT: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
    WEISMAN_KLEMP_EXPERIMENT: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
}


def get_test_data_root_path() -> pathlib.Path:
    test_utils_path = pathlib.Path(__file__).parent
    model_path = test_utils_path.parent
    common_path = model_path.parent.parent.parent.parent
    env_base_path = os.getenv("TEST_DATA_PATH")

    if env_base_path:
        return pathlib.Path(env_base_path)
    else:
        return common_path.parent.joinpath(DEFAULT_TEST_DATA_FOLDER)


TEST_DATA_ROOT = get_test_data_root_path()
SERIALIZED_DATA_PATH = TEST_DATA_ROOT.joinpath("ser_icondata")
GRIDS_PATH = TEST_DATA_ROOT.joinpath("grids")

DATA_URIS = {
    1: "https://polybox.ethz.ch/index.php/s/f42nsmvgOoWZPzi/download",
    2: "https://polybox.ethz.ch/index.php/s/P6F6ZbzWHI881dZ/download",
    4: "https://polybox.ethz.ch/index.php/s/NfES3j9no15A0aX/download",
}
DATA_URIS_APE = {1: "https://polybox.ethz.ch/index.php/s/2n2WpTgZFlTCTHu/download"}
DATA_URIS_JABW = {1: "https://polybox.ethz.ch/index.php/s/5W3Z2K6pyo0egzo/download"}
DATA_URIS_GAUSS3D = {1: "https://polybox.ethz.ch/index.php/s/ZuqDIREPVits9r0/download"}
DATA_URIS_WK = {1: "https://polybox.ethz.ch/index.php/s/ByLnyii7MMRHJbK/download"}


def get_global_grid_params(experiment: str) -> tuple[int, int]:
    """Get the grid root and level from the experiment name.

    Reads the level and root parameters from a string in the canonical ICON gridfile format
        RxyBab where 'xy' and 'ab' are numbers and denote the root and level of the icosahedron grid construction.

        Args: experiment: str: The experiment name.
        Returns: tuple[int, int]: The grid root and level.
    """
    if "torus" in experiment:
        # these magic values seem to mark a torus: they are set in all torus grid files.
        return 0, 2

    try:
        root, level = map(int, re.search("[Rr](\d+)[Bb](\d+)", experiment).groups())
        return root, level
    except AttributeError as err:
        raise ValueError(
            f"Could not parse grid_root and grid_level from experiment: {experiment} no 'rXbY'pattern."
        ) from err


def get_grid_id_for_experiment(experiment) -> uuid.UUID:
    """Get the unique id of the grid used in the experiment.

    These ids are encoded in the original grid file that was used to run the simulation, but not serialized when generating the test data. So we duplicate the information here.

    TODO (@halungge): this becomes obsolete once we get the connectivities from the grid files.
    """
    try:
        return GRID_IDS[experiment]
    except KeyError as err:
        raise ValueError(f"Experiment '{experiment}' has no grid id ") from err


def get_processor_properties_for_run(run_instance):
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path, processor_properties):
    return base_path.absolute().joinpath(f"mpitask{processor_properties.comm_size}")


def get_datapath_for_experiment(ranked_base_path, experiment=REGIONAL_EXPERIMENT):
    return ranked_base_path.joinpath(f"{experiment}/ser_data")


def create_icon_serial_data_provider(
    datapath, processor_props, backend: Optional[gtx_backend.Backend]
) -> serialbox.IconSerialDataProvider:
    # note: this needs to be here, otherwise spack doesn't find serialbox
    from icon4py.model.testing.serialbox import IconSerialDataProvider

    return IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import re
import uuid

from icon4py.model.common.decomposition import definitions as decomposition


DEFAULT_TEST_DATA_FOLDER = "testdata"
GLOBAL_EXPERIMENT = "exclaim_ape_R02B04"
REGIONAL_EXPERIMENT = "mch_ch_r04b09_dsl"
R02B04_GLOBAL = "r02b04_global"
JABW_EXPERIMENT = "jabw_R02B04"
GAUSS3D_EXPERIMENT = "gauss3d_torus"
WEISMAN_KLEMP_EXPERIMENT = "weisman_klemp_torus"

MC_CH_R04B09_DSL_GRID_URI = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
R02B04_GLOBAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/AKAO6ImQdIatnkB/download"
TORUS_100X116_1000M_GRID_URI = "https://polybox.ethz.ch/index.php/s/yqvotFss9i1OKzs/download"
TORUS_GRID_URI = "https://polybox.ethz.ch/index.php/s/CUppakxhyUmof7o/download"


GRID_URIS = {
    REGIONAL_EXPERIMENT: MC_CH_R04B09_DSL_GRID_URI,
    R02B04_GLOBAL: R02B04_GLOBAL_GRID_URI,
    # TODO (Chia Rui): check if we can use the grid for gauss3d and get a good quality data for testing microphysics
    WEISMAN_KLEMP_EXPERIMENT: TORUS_100X116_1000M_GRID_URI,
    GAUSS3D_EXPERIMENT: TORUS_GRID_URI,
}

GRID_IDS = {
    GLOBAL_EXPERIMENT: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    REGIONAL_EXPERIMENT: uuid.UUID("f2e06839-694a-cca1-a3d5-028e0ff326e0"),
    JABW_EXPERIMENT: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    GAUSS3D_EXPERIMENT: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
    WEISMAN_KLEMP_EXPERIMENT: uuid.UUID("24fe2406-8066-11e8-bc5b-f3dc5af69303"),
}


def get_test_data_root_path() -> pathlib.Path:
    test_utils_path = pathlib.Path(__file__).parent
    model_path = test_utils_path.parent.parent
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
    1: "https://polybox.ethz.ch/index.php/s/xhooaubvGffG8Qy/download",
    2: "https://polybox.ethz.ch/index.php/s/P6F6ZbzWHI881dZ/download",
    4: "https://polybox.ethz.ch/index.php/s/NfES3j9no15A0aX/download",
}
DATA_URIS_APE = {1: "https://polybox.ethz.ch/index.php/s/y9WRP1mpPlf2BtM/download"}
DATA_URIS_JABW = {1: "https://polybox.ethz.ch/index.php/s/kp9Rab00guECrEd/download"}
DATA_URIS_GAUSS3D = {1: "https://polybox.ethz.ch/index.php/s/IiRimdJH2ZBZ1od/download"}
DATA_URIS_WK = {1: "https://polybox.ethz.ch/index.php/s/91DEUGmAkBgrXO6/download"}


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


def create_icon_serial_data_provider(datapath, processor_props):
    from icon4py.model.common.test_utils.serialbox_utils import IconSerialDataProvider

    return IconSerialDataProvider(
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )

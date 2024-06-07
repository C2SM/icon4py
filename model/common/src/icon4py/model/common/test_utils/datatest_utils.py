# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import re
from pathlib import Path

from icon4py.model.common.decomposition.definitions import get_processor_properties


DEFAULT_TEST_DATA_FOLDER = "testdata"
GLOBAL_EXPERIMENT = "exclaim_ape_R02B04"
REGIONAL_EXPERIMENT = "mch_ch_r04b09_dsl"
R02B04_GLOBAL = "r02b04_global"
JABW_EXPERIMENT = "jabw_R02B04"

MC_CH_R04B09_DSL_GRID_URI = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
R02B04_GLOBAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/AKAO6ImQdIatnkB/download"
GRID_URIS = {
    REGIONAL_EXPERIMENT: MC_CH_R04B09_DSL_GRID_URI,
    R02B04_GLOBAL: R02B04_GLOBAL_GRID_URI,
}


def get_test_data_root_path() -> Path:
    test_utils_path = Path(__file__).parent
    model_path = test_utils_path.parent.parent
    common_path = model_path.parent.parent.parent.parent
    env_base_path = os.getenv("TEST_DATA_PATH")

    if env_base_path:
        return Path(env_base_path)
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


def get_global_grid_params(experiment: str) -> tuple[int, int]:
    """Get the grid root and level from the experiment name.

    Reads the level and root parameters from a string in the canonical ICON gridfile format
        RxyBab where 'xy' and 'ab' are numbers and denote the root and level of the icosahedron grid construction.

        Args: experiment: str: The experiment name.
        Returns: tuple[int, int]: The grid root and level.
    """
    try:
        root, level = map(int, re.search("[Rr](\d+)[Bb](\d+)", experiment).groups())
        return root, level
    except AttributeError as err:
        raise ValueError(
            f"Could not parse grid_root and grid_level from experiment: {experiment} no 'rXbY'pattern."
        ) from err


def get_processor_properties_for_run(run_instance):
    return get_processor_properties(run_instance)


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

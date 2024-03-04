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
from pathlib import Path

from icon4py.model.common.decomposition.definitions import get_processor_properties


DEFAULT_TEST_DATA_FOLDER = "testdata"


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

DATA_URIS = {
    1: "https://polybox.ethz.ch/index.php/s/y2IMUx4pOQ6lyZ7/download",
    2: "https://polybox.ethz.ch/index.php/s/YyC5qDJWyC39y7u/download",
    4: "https://polybox.ethz.ch/index.php/s/UIHOVJs6FVPpz9V/download",
}
DATA_URIS_APE = {1: "https://polybox.ethz.ch/index.php/s/uK3jtrWK90Z4kHC/download"}
DATA_URIS_JABW = {1: "https://polybox.ethz.ch/index.php/s/kp9Rab00guECrEd/download"}

REGIONAL_EXPERIMENT = "mch_ch_r04b09_dsl"
GLOBAL_EXPERIMENT = "exclaim_ape_R02B04"
JABW_EXPERIMENT = "jabw_R02B04"


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

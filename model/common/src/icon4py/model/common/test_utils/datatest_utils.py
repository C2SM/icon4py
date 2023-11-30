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

from pathlib import Path

from icon4py.model.common.decomposition.definitions import get_processor_properties


TEST_UTILS_PATH = Path(__file__).parent
MODEL_PATH = TEST_UTILS_PATH.parent.parent
COMMON_PATH = MODEL_PATH.parent.parent.parent.parent
BASE_PATH = COMMON_PATH.parent.joinpath("testdata")


# TODO: a run that contains all the fields needed for dycore, diffusion, interpolation fields needs to be consolidated
DATA_URIS = {
    1: "https://polybox.ethz.ch/index.php/s/re46l1xnJZ4uCMx/download",
    2: "https://polybox.ethz.ch/index.php/s/YyC5qDJWyC39y7u/download",
    4: "https://polybox.ethz.ch/index.php/s/UIHOVJs6FVPpz9V/download",
}
DATA_URIS_APE = {1: "https://polybox.ethz.ch/index.php/s/SdlHTlsHCwcn5J5/download"}
SER_DATA_BASEPATH = BASE_PATH.joinpath("ser_icondata")


def get_processor_properties_for_run(run_instance):
    return get_processor_properties(run_instance)


def get_ranked_data_path(base_path, processor_properties):
    return base_path.absolute().joinpath(f"mpitask{processor_properties.comm_size}")


def get_datapath_for_experiment(ranked_base_path, experiment="mch_ch_r04b09_dsl"):
    return ranked_base_path.joinpath(f"{experiment}/ser_data")


def create_icon_serial_data_provider(datapath, processor_props):
    from icon4py.model.common.test_utils.serialbox_utils import IconSerialDataProvider

    return IconSerialDataProvider(
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )

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

import pytest

from .data_handling import download_and_extract
from .parallel_helpers import processor_props  # noqa: F401
from .serialbox_utils import IconSerialDataProvider


test_utils = Path(__file__).parent
model = test_utils.parent.parent
common = model.parent.parent.parent.parent
base_path = common.parent.joinpath("testdata")

data_uris = {
    1: "https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p/download",
    2: "https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK/download",
    4: "https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5/download",
}

ser_data_basepath = base_path.joinpath("ser_icondata")


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):  # noqa: F811 # fixtures
    return ser_data_basepath.absolute().joinpath(f"mpitask{processor_props.comm_size}")


@pytest.fixture(scope="session")
def datapath(ranked_data_path):
    return ranked_data_path.joinpath("mch_ch_r04b09_dsl/ser_data")


@pytest.fixture(scope="session")
def download_ser_data(request, processor_props, ranked_data_path):  # noqa: F811 # fixtures
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    try:
        has_data_marker = any(map(lambda i: i.iter_markers(name="datatest"), request.node.items))
        if not has_data_marker or not request.config.getoption("datatest"):
            pytest.skip("not running datatest marked tests")
    except ValueError:
        pass

    try:
        uri = data_uris[processor_props.comm_size]

        data_file = ranked_data_path.joinpath(
            f"mch_ch_r04b09_dsl_mpitask{processor_props.comm_size}.tar.gz"
        ).name
        if processor_props.rank == 0:
            download_and_extract(uri, ranked_data_path, data_file)
        if processor_props.comm:
            processor_props.comm.barrier()
    except KeyError:
        raise AssertionError(
            f"no data for communicator of size {processor_props.comm_size} exists, use 1, 2 or 4"
        )


@pytest.fixture(scope="session")
def data_provider(
    download_ser_data, datapath, processor_props  # noqa: F811 # fixtures
) -> IconSerialDataProvider:
    return IconSerialDataProvider(
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


@pytest.fixture
def icon_grid(grid_savepoint):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid()


@pytest.fixture
def decomposition_info(data_provider):
    return data_provider.from_savepoint_grid().construct_decomposition_info()


@pytest.fixture
def damping_height():
    return 12500


@pytest.fixture
def ndyn_substeps():
    """
    Return number of dynamical substeps.

    Serialized data uses a reduced number (2 instead of the default 5) in order to reduce the amount
    of data generated.
    """
    return 2


@pytest.fixture
def linit():
    """
    Set the 'linit' flag for the ICON diffusion data savepoint.

    Defaults to False
    """
    return False


@pytest.fixture
def step_date_init():
    """
    Set the step date for the loaded ICON time stamp at start of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def step_date_exit():
    """
    Set the step date for the loaded ICON time stamp at the end of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def interpolation_savepoint(data_provider):  # fixtures
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider):  # fixtures
    """Load data from ICON mestric state savepoint."""
    return data_provider.from_metrics_savepoint()

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

from ..decomposition.definitions import SingleNodeRun, get_processor_properties
from .data_handling import download_and_extract
from .serialbox_utils import IconSerialDataProvider


TEST_UTILS_PATH = Path(__file__).parent
MODEL_PATH = TEST_UTILS_PATH.parent.parent
COMMON_PATH = MODEL_PATH.parent.parent.parent.parent
BASE_PATH = COMMON_PATH.parent.joinpath("testdata")

# TODO: a run that contains all the fields needed for dycore, diffusion, interpolation fields needs to be consolidated
DATA_URIS = {
    1: "https://polybox.ethz.ch/index.php/s/psBNNhng0h9KrB4/download",
    2: "https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK/download",
    4: "https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5/download",
}

SER_DATA_BASEPATH = BASE_PATH.joinpath("ser_icondata")


def get_processor_properties_for_run(run_instance):
    return get_processor_properties(run_instance)


def get_ranked_data_path(base_path, processor_properties):
    return base_path.absolute().joinpath(f"mpitask{processor_properties.comm_size}")


def get_datapath_for_ranked_data(ranked_base_path):
    return ranked_base_path.joinpath("mch_ch_r04b09_dsl/ser_data")


def create_icon_serial_data_provider(datapath, processor_props):
    return IconSerialDataProvider(
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )


@pytest.fixture(params=[False], scope="session")
def processor_props(request):
    return get_processor_properties_for_run(SingleNodeRun())


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):
    return get_ranked_data_path(SER_DATA_BASEPATH, processor_props)


@pytest.fixture(scope="session")
def datapath(ranked_data_path):
    return get_datapath_for_ranked_data(ranked_data_path)


@pytest.fixture(scope="session")
def download_ser_data(request, processor_props, ranked_data_path, pytestconfig):
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
        uri = DATA_URIS[processor_props.comm_size]

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
def data_provider(download_ser_data, datapath, processor_props):
    return create_icon_serial_data_provider(datapath, processor_props)


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
def interpolation_savepoint(data_provider):  # F811
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider):  # F811
    """Load data from ICON mestric state savepoint."""
    return data_provider.from_metrics_savepoint()


@pytest.fixture
def metrics_nonhydro_savepoint(data_provider):  # F811
    """Load data from ICON metric state nonhydro savepoint."""
    return data_provider.from_metrics_nonhydro_savepoint()


@pytest.fixture
def savepoint_velocity_init(data_provider, step_date_init, istep, vn_only, jstep):  # F811
    """
    Load data from ICON savepoint at start of velocity_advection module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_init(
        istep=istep, vn_only=vn_only, date=step_date_init, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_init(data_provider, step_date_init, istep, jstep):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_init(istep=istep, date=step_date_init, jstep=jstep)


@pytest.fixture
def savepoint_velocity_exit(data_provider, step_date_exit, istep, vn_only, jstep):  # F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_exit(
        istep=istep, vn_only=vn_only, date=step_date_exit, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_exit(data_provider, step_date_exit, istep, jstep):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_exit(istep=istep, date=step_date_exit, jstep=jstep)


@pytest.fixture
def savepoint_nonhydro_step_exit(data_provider, step_date_exit, jstep):  # noqa F811
    """
    Load data from ICON savepoint at final exit (after predictor and corrector, and 3 final stencils) of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_step_exit(date=step_date_exit, jstep=jstep)


@pytest.fixture
def istep():
    return 1


@pytest.fixture
def jstep():
    return 0


@pytest.fixture
def ntnd(savepoint_velocity_init):
    return savepoint_velocity_init.get_metadata("ntnd").get("ntnd")


@pytest.fixture
def vn_only():
    return False

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

import pytest

from ..decomposition.definitions import SingleNodeRun
from .data_handling import download_and_extract
from .datatest_utils import (
    DATA_URIS,
    DATA_URIS_APE,
    DATA_URIS_JABW,
    GLOBAL_EXPERIMENT,
    JABW_EXPERIMENT,
    REGIONAL_EXPERIMENT,
    SERIALIZED_DATA_PATH,
    create_icon_serial_data_provider,
    get_datapath_for_experiment,
    get_processor_properties_for_run,
    get_ranked_data_path,
)


@pytest.fixture
def experiment():
    return REGIONAL_EXPERIMENT


@pytest.fixture(params=[False], scope="session")
def processor_props(request):
    return get_processor_properties_for_run(SingleNodeRun())


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):
    return get_ranked_data_path(SERIALIZED_DATA_PATH, processor_props)


@pytest.fixture
def datapath(ranked_data_path, experiment):
    return get_datapath_for_experiment(ranked_data_path, experiment)


@pytest.fixture
def download_ser_data(request, processor_props, ranked_data_path, experiment, pytestconfig):
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    try:
        if not request.config.getoption("datatest"):
            pytest.skip("not running datatest marked test")
    except ValueError:
        pass

    try:
        destination_path = get_datapath_for_experiment(ranked_data_path, experiment)
        if experiment == GLOBAL_EXPERIMENT:
            uri = DATA_URIS_APE[processor_props.comm_size]
        elif experiment == JABW_EXPERIMENT:
            uri = DATA_URIS_JABW[processor_props.comm_size]
        else:
            uri = DATA_URIS[processor_props.comm_size]

        data_file = ranked_data_path.joinpath(
            f"{experiment}_mpitask{processor_props.comm_size}.tar.gz"
        ).name
        if processor_props.rank == 0:
            download_and_extract(uri, ranked_data_path, destination_path, data_file)
        if processor_props.comm:
            processor_props.comm.barrier()
    except KeyError as err:
        raise AssertionError(
            f"no data for communicator of size {processor_props.comm_size} exists, use 1, 2 or 4"
        ) from err


@pytest.fixture
def data_provider(download_ser_data, datapath, processor_props):
    return create_icon_serial_data_provider(datapath, processor_props)


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


def is_regional(experiment_name):
    return experiment_name == REGIONAL_EXPERIMENT


@pytest.fixture
def icon_grid(grid_savepoint, experiment):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid(on_gpu=False)


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
def savepoint_velocity_init(data_provider, step_date_init, istep_init, vn_only, jstep_init):  # F811
    """
    Load data from ICON savepoint at start of velocity_advection module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_init(
        istep=istep_init, vn_only=vn_only, date=step_date_init, jstep=jstep_init
    )


@pytest.fixture
def savepoint_nonhydro_init(data_provider, step_date_init, istep_init, jstep_init):
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_init(
        istep=istep_init, date=step_date_init, jstep=jstep_init
    )


@pytest.fixture
def savepoint_velocity_exit(data_provider, step_date_exit, istep_exit, vn_only, jstep_exit):  # F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_exit(
        istep=istep_exit, vn_only=vn_only, date=step_date_exit, jstep=jstep_exit
    )


@pytest.fixture
def savepoint_nonhydro_exit(data_provider, step_date_exit, istep_exit, jstep_exit):
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_exit(
        istep=istep_exit, date=step_date_exit, jstep=jstep_exit
    )


@pytest.fixture
def savepoint_nonhydro_step_exit(data_provider, step_date_exit, jstep_exit):
    """
    Load data from ICON savepoint at final exit (after predictor and corrector, and 3 final stencils) of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_step_exit(date=step_date_exit, jstep=jstep_exit)


@pytest.fixture
def istep_init():
    return 1


@pytest.fixture
def istep_exit():
    return 1


@pytest.fixture
def jstep_init():
    return 0


@pytest.fixture
def jstep_exit():
    return 0


@pytest.fixture
def ntnd(savepoint_velocity_init):
    return savepoint_velocity_init.get_metadata("ntnd").get("ntnd")


@pytest.fixture
def vn_only():
    return False

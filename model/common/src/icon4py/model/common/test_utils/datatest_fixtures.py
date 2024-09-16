# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.decomposition.definitions as decomposition

from . import data_handling as data, datatest_utils as dt_utils


@pytest.fixture
def experiment():
    return dt_utils.REGIONAL_EXPERIMENT


@pytest.fixture(params=[False], scope="session")
def processor_props(request):
    return dt_utils.get_processor_properties_for_run(decomposition.SingleNodeRun())


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):
    return dt_utils.get_ranked_data_path(dt_utils.SERIALIZED_DATA_PATH, processor_props)


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
        destination_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
        if experiment == dt_utils.GLOBAL_EXPERIMENT:
            uri = dt_utils.DATA_URIS_APE[processor_props.comm_size]
        elif experiment == dt_utils.JABW_EXPERIMENT:
            uri = dt_utils.DATA_URIS_JABW[processor_props.comm_size]
        elif experiment == dt_utils.GAUSS3D_EXPERIMENT:
            uri = dt_utils.DATA_URIS_GAUSS3D[processor_props.comm_size]
        elif experiment == dt_utils.WEISMAN_KLEMP_EXPERIMENT:
            uri = dt_utils.DATA_URIS_WK[processor_props.comm_size]
        else:
            uri = dt_utils.DATA_URIS[processor_props.comm_size]

        data_file = ranked_data_path.joinpath(
            f"{experiment}_mpitask{processor_props.comm_size}.tar.gz"
        ).name
        if processor_props.rank == 0:
            data.download_and_extract(uri, ranked_data_path, destination_path, data_file)
        if processor_props.comm:
            processor_props.comm.barrier()
    except KeyError as err:
        raise AssertionError(
            f"no data for communicator of size {processor_props.comm_size} exists, use 1, 2 or 4"
        ) from err


@pytest.fixture
def data_provider(download_ser_data, ranked_data_path, experiment, processor_props):
    data_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
    return dt_utils.create_icon_serial_data_provider(data_path, processor_props)


@pytest.fixture
def grid_savepoint(data_provider, experiment):
    root, level = dt_utils.get_global_grid_params(experiment)
    grid_id = dt_utils.get_grid_id_for_experiment(experiment)
    return data_provider.from_savepoint_grid(grid_id, root, level)


def is_regional(experiment_name):
    return experiment_name == dt_utils.REGIONAL_EXPERIMENT


@pytest.fixture
def icon_grid(grid_savepoint):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid(on_gpu=False)


@pytest.fixture
def decomposition_info(data_provider, experiment):
    root, level = dt_utils.get_global_grid_params(experiment)
    grid_id = dt_utils.get_grid_id_for_experiment(experiment)
    return data_provider.from_savepoint_grid(
        grid_id=grid_id, grid_root=root, grid_level=level
    ).construct_decomposition_info()


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
def savepoint_diffusion_init(
    data_provider,  # noqa: F811 # imported fixtures data_provider
    linit,  # noqa: F811 # imported fixtures linit
    step_date_init,  # noqa: F811 # imported fixtures data_provider
):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_date_init'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag can be set by overriding the 'linit' fixture
    """
    return data_provider.from_savepoint_diffusion_init(linit=linit, date=step_date_init)


@pytest.fixture
def savepoint_diffusion_exit(
    data_provider,  # noqa: F811 # imported fixtures data_provider`
    linit,  # noqa: F811 # imported fixtures linit`
    step_date_exit,  # noqa: F811 # imported fixtures step_date_exit`
):
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=linit, date=step_date_exit)
    return sp


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


@pytest.fixture
def lowest_layer_thickness(experiment):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        return 20.0
    else:
        return 50.0


@pytest.fixture
def model_top_height(experiment):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        return 23000.0
    elif experiment == dt_utils.GLOBAL_EXPERIMENT:
        return 75000.0
    else:
        return 23500.0


@pytest.fixture
def flat_height():
    return 16000.0


@pytest.fixture
def stretch_factor(experiment):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        return 0.65
    elif experiment == dt_utils.GLOBAL_EXPERIMENT:
        return 0.9
    else:
        return 1.0


@pytest.fixture
def damping_height(experiment):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        return 12500.0
    elif experiment == dt_utils.GLOBAL_EXPERIMENT:
        return 50000.0
    else:
        return 45000.0


@pytest.fixture
def htop_moist_proc():
    return 22500.0


@pytest.fixture
def maximal_layer_thickness():
    return 25000.0


@pytest.fixture
def top_height_limit_for_maximal_layer_thickness():
    return 15000.0

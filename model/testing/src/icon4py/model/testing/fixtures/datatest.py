# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from gt4py.next import backend as gtx_backend

import icon4py.model.common.decomposition.definitions as decomposition
from icon4py.model.common import model_backends
from icon4py.model.common.grid import base as base_grid, simple as simple_grid
from icon4py.model.testing import (
    config,
    data_handling as data,
    datatest_utils as dt_utils,
    definitions,
    locking,
)
from icon4py.model.testing.datatest_utils import GLOBAL_EXPERIMENT, REGIONAL_EXPERIMENT


if TYPE_CHECKING:
    import pathlib

    from icon4py.model.testing import serialbox

DEFAULT_GRID: Final[str] = "simple_grid"
VALID_GRIDS: tuple[str, str, str] = ("simple_grid", "icon_grid", "icon_grid_global")


def _check_backend_validity(backend_name: str) -> None:
    if backend_name not in model_backends.BACKENDS:
        available_backends = ", ".join([f"'{k}'" for k in model_backends.BACKENDS.keys()])
        raise Exception(
            "Need to select a backend. Select from: ["
            + available_backends
            + "] and pass it as an argument to --backend when invoking pytest."
        )


def _check_grid_validity(grid_name: str) -> None:
    assert (
        grid_name in VALID_GRIDS
    ), f"Invalid value for '--grid' option - possible names are {VALID_GRIDS}"


def _get_grid(
    selected_grid_type: str, selected_backend: gtx_backend.Backend | None
) -> base_grid.Grid:
    match selected_grid_type:
        case "icon_grid":
            from icon4py.model.testing.grid_utils import get_grid_manager_for_experiment

            grid_instance = get_grid_manager_for_experiment(
                REGIONAL_EXPERIMENT, keep_skip_values=False, backend=selected_backend
            ).grid
            return grid_instance
        case "icon_grid_global":
            from icon4py.model.testing.grid_utils import get_grid_manager_for_experiment

            grid_instance = get_grid_manager_for_experiment(
                GLOBAL_EXPERIMENT, keep_skip_values=False, backend=selected_backend
            ).grid
            return grid_instance
        case _:
            return simple_grid.simple_grid(selected_backend)


@pytest.fixture(scope="session")
def backend(request):
    try:
        backend_option = request.config.getoption("backend")
    except ValueError:
        backend_option = model_backends.DEFAULT_BACKEND
    else:
        _check_backend_validity(backend_option)

    selected_backend = model_backends.BACKENDS[backend_option]
    return selected_backend


@pytest.fixture(scope="session")
def grid(request, backend):
    try:
        grid_option = request.config.getoption("grid")
    except ValueError:
        grid_option = DEFAULT_GRID
    else:
        _check_grid_validity(grid_option)
    grid = _get_grid(grid_option, backend)
    return grid


@pytest.fixture
def experiment():
    return dt_utils.REGIONAL_EXPERIMENT


@pytest.fixture(scope="session", params=[False])
def processor_props(request):
    with_mpi = request.param
    runtype = decomposition.get_runtype(with_mpi=with_mpi)
    yield decomposition.get_processor_properties(runtype)


@pytest.fixture(scope="session")
def ranked_data_path(processor_props: decomposition.ProcessProperties) -> pathlib.Path:
    return dt_utils.get_ranked_data_path(
        definitions.serialized_data_path(), processor_props.comm_size
    )


def _download_ser_data(
    comm_size: int,
    _ranked_data_path: pathlib.Path,
    _experiment: str,
):
    # not a fixture to be able to use this function outside of pytest
    try:
        destination_path = dt_utils.get_datapath_for_experiment(_ranked_data_path, _experiment)
        if _experiment == dt_utils.GLOBAL_EXPERIMENT:
            uri = dt_utils.DATA_URIS_APE[comm_size]
        elif _experiment == dt_utils.JABW_EXPERIMENT:
            uri = dt_utils.DATA_URIS_JABW[comm_size]
        elif _experiment == dt_utils.GAUSS3D_EXPERIMENT:
            uri = dt_utils.DATA_URIS_GAUSS3D[comm_size]
        elif _experiment == dt_utils.WEISMAN_KLEMP_EXPERIMENT:
            uri = dt_utils.DATA_URIS_WK[comm_size]
        else:
            uri = dt_utils.DATA_URIS[comm_size]

        data_file = _ranked_data_path.joinpath(f"{_experiment}_mpitask{comm_size}.tar.gz").name
        _ranked_data_path.mkdir(parents=True, exist_ok=True)
        if config.ENABLE_TESTDATA_DOWNLOAD:
            with locking.lock(_ranked_data_path):
                # Note: if the lock would be created for `destination_path` it would always exist...
                if not destination_path.exists():
                    data.download_and_extract(uri, _ranked_data_path, data_file)
        else:
            # If test data download is disabled, we check if the directory exists
            # without locking. We assume the location is managed by the user
            # and avoid locking shared directories (e.g. on CI).
            if not destination_path.exists():
                raise RuntimeError(
                    f"Serialization data {data_file} does not exist, and downloading is disabled."
                )
    except KeyError as err:
        raise RuntimeError(
            f"No data for communicator of size {comm_size} exists, use 1, 2 or 4"
        ) from err


@pytest.fixture
def download_ser_data(
    request,
    processor_props: decomposition.ProcessProperties,
    ranked_data_path: pathlib.Path,
    experiment: str,
    pytestconfig,
):
    """
    Get the binary ICON data from a remote server.

    Fixture which is a prerequisite of all the other fixtures in this file.
    """
    # we don't want to run this ever if we are not running datatests
    if "not datatest" in request.config.getoption("-k", ""):
        return

    _download_ser_data(processor_props.comm_size, ranked_data_path, experiment)


@pytest.fixture
def data_provider(
    download_ser_data, ranked_data_path, experiment, processor_props, backend
) -> serialbox.IconSerialDataProvider:
    data_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
    return dt_utils.create_icon_serial_data_provider(data_path, processor_props, backend)


@pytest.fixture
def grid_savepoint(
    data_provider: serialbox.IconSerialDataProvider, experiment: str
) -> serialbox.IconGridSavepoint:
    root, level = dt_utils.get_global_grid_params(experiment)
    grid_id = dt_utils.get_grid_id_for_experiment(experiment)
    return data_provider.from_savepoint_grid(grid_id, root, level)


def is_regional(experiment_name):
    return experiment_name == dt_utils.REGIONAL_EXPERIMENT


@pytest.fixture
def icon_grid(
    grid_savepoint: serialbox.IconGridSavepoint, backend: gtx_backend.Backend
) -> base_grid.Grid:
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid(keep_skip_values=False, backend=backend)


@pytest.fixture
def decomposition_info(data_provider:serialbox.IconSerialDataProvider, experiment:str)->definitions.DecompositionInfo:
    root, level = dt_utils.get_global_grid_params(experiment)
    grid_id = dt_utils.get_grid_id_for_experiment(experiment)
    return data_provider.from_savepoint_grid(
        grid_id=grid_id, grid_root=root, grid_level=level
    ).construct_decomposition_info()


@pytest.fixture
def ndyn_substeps(experiment) -> int:
    """
    Return number of dynamical substeps.

    Serialized data of global and regional experiments uses a reduced number
    (2 instead of the default 5) in order to reduce the amount of data generated.
    """
    if experiment == dt_utils.GAUSS3D_EXPERIMENT:
        return 5
    else:
        return 2


@pytest.fixture
def linit() -> bool:
    """
    Set the 'linit' flag for the ICON diffusion data savepoint.

    Defaults to False
    """
    return False


@pytest.fixture
def step_date_init() -> str:
    """
    Set the step date for the loaded ICON time stamp at start of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def substep_init() -> int:
    return 1


@pytest.fixture
def substep_exit() -> int:
    return 1


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
    """Load data from ICON metric state savepoint."""
    return data_provider.from_metrics_savepoint()


@pytest.fixture
def metrics_nonhydro_savepoint(data_provider):  # F811
    """Load data from ICON metric state nonhydro savepoint."""
    return data_provider.from_metrics_nonhydro_savepoint()


@pytest.fixture
def topography_savepoint(data_provider):  # F811
    """Load data from ICON external parameters savepoint."""
    return data_provider.from_topography_savepoint()


@pytest.fixture
def savepoint_velocity_init(data_provider, step_date_init, istep_init, substep_init):  # F811
    """
    Load data from ICON savepoint at start of subroutine velocity_tendencies in mo_velocity_advection.f90.

    metadata to select a unique savepoint:
    - date: <iso_string> of the simulation timestep
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_velocity_init(
        istep=istep_init, date=step_date_init, substep=substep_init
    )


@pytest.fixture
def savepoint_nonhydro_init(
    data_provider: serialbox.IconSerialDataProvider,
    step_date_init: str,
    istep_init: int,
    substep_init: int,
) -> serialbox.IconNonHydroInitSavepoint:
    """
    Load data from ICON savepoint at init of subroutine nh_solve in mo_solve_nonhydro.f90 of solve_nonhydro module.

     metadata to select a unique savepoint:
    - date: <iso_string> of the simulation timestep
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_nonhydro_init(
        istep=istep_init, date=step_date_init, substep=substep_init
    )


@pytest.fixture
def savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init(
    data_provider, istep_init, step_date_init, substep_init
):
    """
    Load data from ICON savepoint before edge diagnostics computations and update of new vn
    (formally known as stencils 15 to 28) in mo_solve_nonhydro.f90 of solve_nonhydro module.

     metadata to select a unique savepoint:
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - date: <iso_string> of the simulation timestep
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init(
        istep=istep_init, date=step_date_init, substep=substep_init
    )


@pytest.fixture
def savepoint_vertically_implicit_dycore_solver_init(
    data_provider, istep_init, step_date_init, substep_init
):
    """
    Load data from ICON savepoint at init of subroutine nh_solve in mo_solve_nonhydro.f90 of solve_nonhydro module.

     metadata to select a unique savepoint:
    - date: <iso_string> of the simulation timestep
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_vertically_implicit_dycore_solver_init(
        istep=istep_init, date=step_date_init, substep=substep_init
    )


@pytest.fixture
def savepoint_velocity_exit(data_provider, step_date_exit, istep_exit, substep_exit):  # F811
    """
    Load data from ICON savepoint at start of subroutine velocity_tendencies in mo_velocity_advection.f90.

    metadata to select a unique savepoint:
    - date: <iso_string> of the simulation timestep
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_velocity_exit(
        istep=istep_exit, date=step_date_exit, substep=substep_exit
    )


@pytest.fixture
def savepoint_nonhydro_exit(data_provider, step_date_exit, istep_exit, substep_exit):
    """
    Load data from ICON savepoint at the end of either predictor or corrector step (istep loop) of
    subroutine nh_solve in mo_solve_nonhydro.f90.

    metadata to select a unique savepoint:
    - date: <iso_string> of the simulation timestep
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_nonhydro_exit(
        istep=istep_exit, date=step_date_exit, substep=substep_exit
    )


@pytest.fixture
def savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit(
    data_provider, istep_exit, step_date_exit, substep_exit
):
    """
    Load data from ICON savepoint at the end of edge diagnostics computations and update of new vn
    (formally known as stencils 15 to 28) in mo_solve_nonhydro.f90.

    metadata to select a unique savepoint:
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - date: <iso_string> of the simulation timestep
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit(
        istep=istep_exit, date=step_date_exit, substep=substep_exit
    )


@pytest.fixture
def savepoint_nonhydro_step_final(data_provider, step_date_exit, substep_exit):
    """
    Load data from ICON savepoint at final exit of subroutine nh_solve in mo_solve_nonhydro.f90.
    (after predictor and corrector and 3 final stencils have run).

     metadata to select a unique savepoint:
    - date: <iso_string> of the simulation timestep
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_nonhydro_step_final(
        date=step_date_exit, substep=substep_exit
    )


@pytest.fixture
def savepoint_diffusion_init(
    data_provider,
    linit,
    step_date_init,
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
    data_provider,  # imported fixtures data_provider`
    linit,  # imported fixtures linit`
    step_date_exit,  # imported fixtures step_date_exit`
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

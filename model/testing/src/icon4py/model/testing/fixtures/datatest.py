# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pkgutil
from typing import TYPE_CHECKING

import gt4py.next.typing as gtx_typing
import pytest

import icon4py.model.common.decomposition.definitions as decomposition
from icon4py.model.common import model_backends, model_options
from icon4py.model.common.grid import base as base_grid
from icon4py.model.testing import (
    config,
    data_handling as data,
    datatest_utils as dt_utils,
    definitions,
    locking,
)


if TYPE_CHECKING:
    import pathlib

    from icon4py.model.testing import serialbox


def _get_backend_like(spec: str) -> model_backends.BackendLike:
    if spec.count(":") > 1:
        raise ValueError(
            "Invalid backend spec in '--backend' option (spec: <backend_name> or <path.to.module>:<symbol>)"
        )

    if ":" in spec:
        backend_like = pkgutil.resolve_name(spec)
    elif spec in model_backends.BACKENDS:
        backend_like = model_backends.BACKENDS[spec]
    else:
        raise ValueError(
            f"Invalid backend name in '--backend' option. It should be one of {[*model_backends.BACKENDS.keys()]}"
        )

    return backend_like


@pytest.fixture(scope="session")
def backend_like(request: pytest.FixtureRequest) -> model_backends.BackendLike:
    """
    Fixture to provide a GT4Py backend or an ICON4Py BackendDescriptor for the tests.

    See `backend` fixture for details.
    """
    spec = request.config.getoption("backend", model_backends.DEFAULT_BACKEND)
    assert isinstance(spec, str), "Backend spec must be a string"
    return _get_backend_like(spec)


@pytest.fixture(scope="session")
def backend(request: pytest.FixtureRequest) -> gtx_typing.Backend | None:
    """
    Fixture to provide a GT4Py backend for the tests.

    The provided backend is instanciated according to the `--backend` pytest
    command line option value, which might refer to a known backend name, or to
    an gt4py backend instance defined in an arbitrary location, by using the
    notation `path.to.module:backend_symbol`.
    """
    # TODO(havogt): eventually all tests should support `backend_like`,
    # then `backend_like` should probably be renamed to `backend`.

    spec = request.config.getoption("backend", model_backends.DEFAULT_BACKEND)
    assert isinstance(spec, str), "Backend spec must be a string"
    backend_like = _get_backend_like(spec)
    # We create a generic concrete backend (no program specific customization).
    return model_options.customize_backend(None, backend_like)


@pytest.fixture
def cpu_allocator() -> gtx_typing.FieldBufferAllocationUtil:
    return model_backends.get_allocator(None)


@pytest.fixture(
    params=[definitions.Experiments.MCH_CH_R04B09, definitions.Experiments.EXCLAIM_APE],
    ids=lambda r: r.name,
)
def experiment(request: pytest.FixtureRequest) -> definitions.Experiment:
    """Default parametrization for experiments.

    The default parametrization is often overwritten for specific tests."""
    return request.param


@pytest.fixture(scope="session", params=[False])
def processor_props(request: pytest.FixtureRequest) -> decomposition.ProcessProperties:
    with_mpi = request.param
    runtype = decomposition.get_runtype(with_mpi=with_mpi)
    return decomposition.get_processor_properties(runtype)


@pytest.fixture(scope="session")
def ranked_data_path(processor_props: decomposition.ProcessProperties) -> pathlib.Path:
    return dt_utils.get_ranked_data_path(
        definitions.serialized_data_path(), processor_props.comm_size
    )


def _download_ser_data(
    comm_size: int,
    _ranked_data_path: pathlib.Path,
    _experiment: definitions.Experiment,
) -> None:
    # not a fixture to be able to use this function outside of pytest
    try:
        destination_path = dt_utils.get_datapath_for_experiment(_ranked_data_path, _experiment)
        uri = _experiment.partitioned_data[comm_size]

        data_file = _ranked_data_path.joinpath(f"{_experiment.name}_mpitask{comm_size}.tar.gz").name
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
    request: pytest.FixtureRequest,
    processor_props: decomposition.ProcessProperties,
    ranked_data_path: pathlib.Path,
    experiment: definitions.Experiment,
    pytestconfig: pytest.Config,
) -> None:
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
    download_ser_data: None,  # downloads data as side-effect
    ranked_data_path: pathlib.Path,
    experiment: definitions.Experiment,
    processor_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend,
) -> serialbox.IconSerialDataProvider:
    data_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
    return dt_utils.create_icon_serial_data_provider(data_path, processor_props.rank, backend)


@pytest.fixture
def grid_savepoint(
    data_provider: serialbox.IconSerialDataProvider, experiment: definitions.Experiment
) -> serialbox.IconGridSavepoint:
    grid_shape = dt_utils.guess_grid_shape(experiment)
    return data_provider.from_savepoint_grid(experiment.name, grid_shape)


@pytest.fixture
def icon_grid(
    grid_savepoint: serialbox.IconGridSavepoint, backend: gtx_typing.Backend
) -> base_grid.Grid:
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid(keep_skip_values=False, backend=backend)


@pytest.fixture
def decomposition_info(
    data_provider: serialbox.IconSerialDataProvider, experiment: definitions.Experiment
) -> decomposition.DecompositionInfo:
    grid_shape = dt_utils.guess_grid_shape(experiment)
    return data_provider.from_savepoint_grid(
        grid_id=experiment.name, grid_shape=grid_shape
    ).construct_decomposition_info()


@pytest.fixture
def ndyn_substeps(experiment: definitions.Experiment) -> int:
    """
    Return number of dynamical substeps.

    Serialized data of global and regional experiments uses a reduced number
    (2 instead of the default 5) in order to reduce the amount of data generated.
    """
    if experiment == definitions.Experiments.GAUSS3D:
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
def step_date_exit() -> str:
    """
    Set the step date for the loaded ICON time stamp at the end of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def interpolation_savepoint(
    data_provider: serialbox.IconSerialDataProvider,
) -> serialbox.InterpolationSavepoint:
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider: serialbox.IconSerialDataProvider) -> serialbox.MetricSavepoint:
    """Load data from ICON metric state savepoint."""
    return data_provider.from_metrics_savepoint()


@pytest.fixture
def topography_savepoint(
    data_provider: serialbox.IconSerialDataProvider,
) -> serialbox.TopographySavepoint:
    """Load data from ICON external parameters savepoint."""
    return data_provider.from_topography_savepoint()


@pytest.fixture
def savepoint_velocity_init(
    data_provider: serialbox.IconSerialDataProvider,
    step_date_init: str,
    istep_init: int,
    substep_init: int,
) -> serialbox.IconVelocityInitSavepoint:
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
def savepoint_dycore_30_to_38_init(
    data_provider: serialbox.IconSerialDataProvider,
    istep_init: int,
    step_date_init: str,
    substep_init: int,
) -> serialbox.IconDycoreInit30To38Savepoint:
    """
    Load data from ICON savepoint directly before the first stencil in
    stencils 30 to 38 in mo_solve_nonhydro.f90 of solve_nonhydro module.
    metadata to select a unique savepoint:
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - date: <iso_string> of the simulation timestep
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_30_to_38_init(
        istep=istep_init, date=step_date_init, substep=substep_init
    )


@pytest.fixture
def savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init(
    data_provider: serialbox.IconSerialDataProvider,
    istep_init: int,
    step_date_init: str,
    substep_init: int,
) -> serialbox.NonHydroInitEdgeDiagnosticsUpdateVnSavepoint:
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
    data_provider: serialbox.IconSerialDataProvider,
    istep_init: int,
    step_date_init: str,
    substep_init: int,
) -> serialbox.NonHydroInitVerticallyImplicitSolverSavepoint:
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
def savepoint_velocity_exit(
    data_provider: serialbox.IconSerialDataProvider,
    step_date_exit: str,
    istep_exit: int,
    substep_exit: int,
) -> serialbox.IconVelocityExitSavepoint:
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
def savepoint_nonhydro_exit(
    data_provider: serialbox.IconSerialDataProvider,
    step_date_exit: str,
    istep_exit: int,
    substep_exit: int,
) -> serialbox.IconNonHydroExitSavepoint:
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
def savepoint_dycore_30_to_38_exit(
    data_provider: serialbox.IconSerialDataProvider,
    istep_exit: int,
    step_date_exit: str,
    substep_exit: int,
) -> serialbox.IconDycoreExit30To38Savepoint:
    """
    Load data from ICON savepoint directly after the last stencil in
    stencils 30 to 38 in mo_solve_nonhydro.f90 of solve_nonhydro module.
    metadata to select a unique savepoint:
    - istep: one of 1 ~ predictor, 2 ~ corrector of dycore integration scheme
    - date: <iso_string> of the simulation timestep
    - substep: dynamical substep
    """
    return data_provider.from_savepoint_30_to_38_exit(
        istep=istep_exit, date=step_date_exit, substep=substep_exit
    )


@pytest.fixture
def savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit(
    data_provider: serialbox.IconSerialDataProvider,
    istep_exit: int,
    step_date_exit: str,
    substep_exit: int,
) -> serialbox.NonHydroExitEdgeDiagnosticsUpdateVnSavepoint:
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
def savepoint_nonhydro_step_final(
    data_provider: serialbox.IconSerialDataProvider,
    step_date_exit: str,
    substep_exit: int,
) -> serialbox.IconNonHydroFinalSavepoint:
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
    data_provider: serialbox.IconSerialDataProvider,
    linit: bool,
    step_date_init: str,
) -> serialbox.IconDiffusionInitSavepoint:
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_date_init'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag can be set by overriding the 'linit' fixture
    """
    return data_provider.from_savepoint_diffusion_init(linit=linit, date=step_date_init)


@pytest.fixture
def savepoint_diffusion_exit(
    data_provider: serialbox.IconSerialDataProvider,
    linit: bool,
    step_date_exit: str,
) -> serialbox.IconDiffusionExitSavepoint:
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=linit, date=step_date_exit)
    return sp


@pytest.fixture
def istep_init() -> int:
    return 1


@pytest.fixture
def istep_exit() -> int:
    return 1


@pytest.fixture
def lowest_layer_thickness(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.MCH_CH_R04B09:
        return 20.0
    else:
        return 50.0


@pytest.fixture
def model_top_height(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.MCH_CH_R04B09:
        return 23000.0
    elif experiment == definitions.Experiments.EXCLAIM_APE:
        return 75000.0
    else:
        return 23500.0


@pytest.fixture
def flat_height() -> float:
    return 16000.0


@pytest.fixture
def stretch_factor(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.MCH_CH_R04B09:
        return 0.65
    elif experiment == definitions.Experiments.EXCLAIM_APE:
        return 0.9
    else:
        return 1.0


@pytest.fixture
def damping_height(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.MCH_CH_R04B09:
        return 12500.0
    elif experiment == definitions.Experiments.EXCLAIM_APE:
        return 50000.0
    else:
        return 45000.0


@pytest.fixture
def htop_moist_proc() -> float:
    return 22500.0


@pytest.fixture
def maximal_layer_thickness() -> float:
    return 25000.0


@pytest.fixture
def rayleigh_coeff(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.EXCLAIM_APE:
        return 0.1
    else:
        return 5.0


@pytest.fixture
def exner_expol(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.EXCLAIM_APE:
        return 0.3333333333333
    else:
        return 0.333


@pytest.fixture
def vwind_offctr(experiment: definitions.Experiment) -> float:
    if experiment == definitions.Experiments.EXCLAIM_APE:
        return 0.15
    else:
        return 0.2


@pytest.fixture
def top_height_limit_for_maximal_layer_thickness() -> float:
    return 15000.0

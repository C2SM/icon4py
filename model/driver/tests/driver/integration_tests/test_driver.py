# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
)

from ..fixtures import *  # noqa: F403


# Tolerances (atol, rtol) per experiment, measured across the CSCS CI backends
_TOLERANCES: dict[test_defs.ExperimentDescription, dict[str, tuple[float, float]]] = {
    test_defs.Experiments.JW: {
        "vn": (5.3e-7, 0.0),
        "w": (8e-9, 0.0),
        "exner": (4.5e-11, 5.5e-11),
        "theta_v": (5.5e-8, 1.3e-10),
        "rho": (1.5e-10, 2.2e-10),
    },
    test_defs.Experiments.GAUSS3D: {
        "vn": (4.1e-13, 0.0),
        "w": (8.1e-14, 0.0),
        "exner": (1.3e-10, 1.3e-10),
        "theta_v": (9.3e-8, 3.1e-10),
        "rho": (1.8e-15, 3.7e-15),
    },
    test_defs.Experiments.MCH_CH_R04B09: {
        "vn": (3.5e-3, 0.0),
        "w": (1e-3, 0.0),
        "exner": (6.8e-7, 9.9e-7),
        "theta_v": (1.2e-3, 3.6e-6),
        "rho": (3.5e-6, 3.7e-6),
    },
    test_defs.Experiments.EXCLAIM_APE_AES: {
        "vn": (6e-7, 0.0),
        "w": (1e-8, 0.0),
        "rho": (9e-10, 0.0),
        "exner": (1e-8, 0.0),
        "theta_v": (0.0, 3e-8),
        "qv": (1e-8, 0.0),
        "qc": (1e-10, 0.0),
        "qr": (1e-10, 0.0),
        "qs": (1e-10, 0.0),
        "qi": (1e-10, 0.0),
        "qg": (1e-10, 0.0),
    },
}


# Metadata selecting the MCH mid-time-step dynamics savepoints (see the MCH branch in
# the test body): solve-nonhydro exit at the corrector (istep=2) of the last substep
# (2 for MCH), and the non-initial diffusion savepoint. Only instantiated for MCH.
@pytest.fixture  # type: ignore[no-redef]  # deliberately shadows the fixtures.py import
def istep_exit() -> int:
    return 2


@pytest.fixture
def substep_exit() -> int:
    return 2


@pytest.fixture
def timeloop_diffusion_linit_exit() -> bool:
    return False


@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "experiment_description, timeloop_date_init, timeloop_date_exit, step_date_exit",
    [
        (
            test_defs.Experiments.JW,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
        ),
        (
            test_defs.Experiments.GAUSS3D,
            "2001-01-01T00:00:00.000",
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
        ),
        (
            test_defs.Experiments.EXCLAIM_APE_AES,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:00.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
    ],
)
def test_driver(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    *,
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    savepoint_time_step_exit: sb.IconTimeStepExitSavepoint,
) -> None:
    """End-to-end standalone-driver validation over one time step.

    Experiments validate the final prognostic state against the end-of-time-step
    (``time-step-exit``) savepoint. EXCLAIM_APE_AES additionally runs muphys and also
    validates the tracers. Exception: MCH_CH_R04B09 compares against the mid-time-step
    dynamics savepoints, because its reference runs NWP physics + limited-area nudging
    after the dynamics, which the driver does not (see the comment in the body).
    Per-field tolerances live in ``_TOLERANCES``.

    muphys (EXCLAIM_APE_AES): runs ``MuphysScheme.AES_GRAUPEL`` -- the port of the exact
    ICON formulation that generated the reference. Graupel is the only *physics*
    parameterization active, so vn/w/rho/exner/theta_v compare tightly; the tracer
    comparison carries residuals from gaps not yet ported:

    - exner / theta_v: recomputed via the exact EOS in ``scatter_to_prognostic``, mirroring
      ICON's phy2dyn coupling (mo_interface_iconam_aes.f90). Measured on v6: exner ~3e-9
      (atol=1e-8), theta_v ~7e-9 relative (rtol=3e-8) -- essentially exact.
    - tracer transport: the driver runs MIURA/PPM advection on the dycore-accumulated
      mass fluxes and airmass, matching the reference configuration (ltransport=.TRUE.),
      so this validates transport+muphys. Measured on v6 (gtfn_cpu): qc/qr/qs/qi/qg are
      bit-exact and qv's residual is ~9e-10 (atol=1e-8) -- the remaining gap stems from
      the clipping / vertical-extent items below.
    - negative tracers: ICON clips them (iqneg_d2p/iqneg_p2d); the driver does not.
    - vertical extent: ICON runs graupel on jks_cloudy..nlev; muphys runs the full column.

    The muphys granule itself is validated in isolation against the aes-graupel savepoints
    in test_muphys_datatest.py.
    """
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            # 'start_of_simulation' stays at the beginning of the experiment: the second
            # MCH_CH_R04B09 case starts the time loop later, i.e. it restarts.
            "start_of_timestepping": datetime.datetime.fromisoformat(timeloop_date_init).replace(
                tzinfo=datetime.UTC
            ),
            "end_of_simulation": datetime.datetime.fromisoformat(timeloop_date_exit).replace(
                tzinfo=datetime.UTC
            ),
        }
    )

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    ds, _ = driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    prognostics = ds.prognostics.current

    computed = {
        "vn": prognostics.vn,
        "w": prognostics.w,
        "rho": prognostics.rho,
        "exner": prognostics.exner,
        "theta_v": prognostics.theta_v,
    }
    if experiment_description is test_defs.Experiments.MCH_CH_R04B09:
        # The MCH reference runs the full NWP physics suite (nwp_phy_nml: convection,
        # radiation, SSO, graupel, satad) plus limited-area boundary nudging AFTER the
        # dynamics -- none of which the driver runs -- so its end-of-step state is not
        # comparable (vn differs by O(10) m/s). Validate against the mid-time-step
        # dynamics savepoints instead until NWP physics is ported.
        diffusion_exit = request.getfixturevalue("savepoint_diffusion_exit")
        nonhydro_exit = request.getfixturevalue("savepoint_nonhydro_exit")
        references = {
            "vn": diffusion_exit.vn(),
            "w": diffusion_exit.w(),
            "rho": nonhydro_exit.rho_new(),
            "exner": diffusion_exit.exner(),
            "theta_v": diffusion_exit.theta_v(),
        }
    else:
        # Nothing runs after diffusion for JW/GAUSS3D, and for EXCLAIM_APE_AES muphys
        # is the only active physics: validate against the end-of-time-step savepoint.
        references = {
            "vn": savepoint_time_step_exit.vn(),
            "w": savepoint_time_step_exit.w(),
            "rho": savepoint_time_step_exit.rho(),
            "exner": savepoint_time_step_exit.exner(),
            "theta_v": savepoint_time_step_exit.theta_v(),
        }

    for tracer in ds.tracers.current.active_fields():
        computed[tracer.name] = tracer.field
        references[tracer.name] = getattr(savepoint_time_step_exit, tracer.name)()

    tolerances = _TOLERANCES[experiment_description]

    for name, reference in references.items():
        atol, rtol = tolerances[name]
        test_utils.assert_dallclose(
            computed[name].asnumpy(),
            reference.asnumpy(),
            atol=atol,
            rtol=rtol,
            err_msg=name,
        )

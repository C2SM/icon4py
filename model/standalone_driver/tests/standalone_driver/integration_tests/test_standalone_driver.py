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
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
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
    # EXCLAIM_APE_AES runs muphys and is validated against the end-of-time-step
    # savepoint (see the test docstring). Fields graupel does not touch (vn/w/rho) use
    # the dynamics tolerances; exner/theta_v are the measured coupling tolerances; qv
    # carries a small transport-off residual while qc/qr/qs/qi/qg match bit-for-bit.
    test_defs.Experiments.EXCLAIM_APE_AES: {
        "vn": (6e-7, 0.0),
        "w": (1e-8, 0.0),
        "rho": (9e-10, 0.0),
        "exner": (1e-8, 0.0),
        "theta_v": (0.0, 3e-8),
        "qv": (2e-6, 0.0),
        "qc": (1e-10, 0.0),
        "qr": (1e-10, 0.0),
        "qs": (1e-10, 0.0),
        "qi": (1e-10, 0.0),
        "qg": (1e-10, 0.0),
    },
}


@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, istep_exit, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
    [
        (
            test_defs.Experiments.JW,
            2,
            5,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
            False,
            False,
        ),
        (
            test_defs.Experiments.GAUSS3D,
            2,
            5,
            "2001-01-01T00:00:00.000",
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
            False,
            False,
        ),
        (
            test_defs.Experiments.EXCLAIM_APE_AES,
            2,
            5,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
            False,
            False,
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            2,
            2,
            "2021-06-20T12:00:00.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            True,
            False,
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            2,
            2,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            False,
            False,
        ),
    ],
)
def test_standalone_driver(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    substep_exit: int,
    savepoint_diffusion_exit: sb.IconDiffusionExitSavepoint,
) -> None:
    """End-to-end standalone-driver validation over one large time step.

    Dynamics-only experiments (JW, GAUSS3D, MCH_CH_R04B09) validate the final prognostic
    state against the mid-time-step dynamics savepoints. EXCLAIM_APE_AES additionally runs
    muphys and validates the full prognostic state (incl. tracers) against the
    end-of-time-step (``time-step-exit``) savepoint. Per-field tolerances live in
    ``_TOLERANCES``.

    muphys (EXCLAIM_APE_AES): runs ``MuphysScheme.AES_GRAUPEL`` -- the port of the exact
    icon-nwp formulation that generated the reference. Graupel is the only *physics*
    parameterization active, so vn/w/rho/exner/theta_v compare tightly; the tracer
    comparison carries residuals from gaps not yet ported:

    - exner / theta_v: recomputed via the exact EOS in ``scatter_to_prognostic``, mirroring
      ICON's phy2dyn coupling (mo_interface_iconam_aes.f90). Measured on v6: exner ~3e-9
      (atol=1e-8), theta_v ~7e-9 relative (rtol=3e-8) -- essentially exact.
    - tracer transport (KNOWN MISMATCH): the reference ran with tracer advection ON
      (ltransport=.TRUE., MIURA/PPM), but the driver disables it here -- it can't yet
      compute airmass (rho*ddqz) or wire live mass fluxes (TODO(OngChia)), so advection
      would divide by a zero airmass. So this validates muphys in isolation, not
      transport+muphys. Over one 300 s step the advective change is small: qv still passes
      at atol=2e-6, qc/qr/qs/qi/qg match bit-for-bit (atol=1e-10).
      TODO (Yilu): revisit once the driver computes airmass + wires the mass fluxes.
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
    ds, _ = standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    prognostics = ds.prognostics.current

    if config.muphys is None:
        # Dynamics-only experiments: validate against the mid-time-step dynamics savepoints.
        computed = {
            "vn": prognostics.vn,
            "w": prognostics.w,
            "exner": prognostics.exner,
            "theta_v": prognostics.theta_v,
            "rho": prognostics.rho,
        }
        references = {
            "vn": savepoint_diffusion_exit.vn(),
            "w": savepoint_diffusion_exit.w(),
            "exner": savepoint_diffusion_exit.exner(),
            "theta_v": savepoint_diffusion_exit.theta_v(),
            "rho": savepoint_nonhydro_exit.rho_new(),
        }
    else:
        # Physics enabled (EXCLAIM_APE_AES): validate the full prognostic state, including
        # tracers, against the end-of-time-step savepoint. Fetched lazily so the dynamics-only
        # experiments (which may lack a time-step-exit savepoint) are unaffected.
        time_step_exit = request.getfixturevalue("savepoint_time_step_exit")
        computed = {
            "vn": prognostics.vn,
            "w": prognostics.w,
            "rho": prognostics.rho,
            "exner": prognostics.exner,
            "theta_v": prognostics.theta_v,
        }
        references = {
            "vn": time_step_exit.vn(),
            "w": time_step_exit.w(),
            "rho": time_step_exit.rho(),
            "exner": time_step_exit.exner(),
            "theta_v": time_step_exit.theta_v(),
        }
        for tracer in ("qv", "qc", "qr", "qs", "qi", "qg"):
            field = getattr(prognostics.tracer, tracer)
            assert field is not None, f"tracer {tracer} must be active for the APE_aes experiment"
            computed[tracer] = field
            references[tracer] = getattr(time_step_exit, tracer)()

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

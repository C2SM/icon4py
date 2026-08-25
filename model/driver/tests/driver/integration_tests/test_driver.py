# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import pathlib
from typing import Callable

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx as tmx_module
from icon4py.model.common import model_backends
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
    config as test_config,
)

from ..fixtures import *  # noqa: F403


def test_tmx_plumbing() -> None:
    """Config-layer TMX wiring: ExperimentConfig.tmx exists, defaults to None, TmxConfig constructs."""
    fields = {f.name: f for f in dataclasses.fields(driver_config.ExperimentConfig)}
    assert "tmx" in fields, "ExperimentConfig must have a 'tmx' field"
    assert fields["tmx"].default is None, "ExperimentConfig.tmx must default to None"
    assert tmx_module.TmxConfig() is not None


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
    # Measured 2026-08-20 on the v08 reference (graupel + tmx, parallel two-layer
    # coupling, zero surface fluxes) with ~2x headroom over the observed max diffs
    # (vn 5.5e-7, w 8.4e-9, exner 5.5e-4, theta_v rel 1.4e-3, qv 5.2e-5, qc 5.0e-5,
    # qi 2.9e-5, qr 1.5e-13, rho 1.6e-10, qs/qg bitwise exact).
    test_defs.Experiments.EXCLAIM_APE_AES: {
        "vn": (1.2e-6, 0.0),
        "w": (2e-8, 0.0),
        "rho": (9e-10, 0.0),
        "exner": (1.2e-3, 0.0),
        "theta_v": (0.0, 3e-3),
        "qv": (1.2e-4, 0.0),
        "qc": (1.2e-4, 0.0),
        "qr": (5e-13, 0.0),
        "qs": (1e-10, 0.0),
        "qi": (6e-5, 0.0),
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
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, timeloop_date_init, timeloop_date_exit, step_date_exit",
    [
        # (
        #     test_defs.Experiments.JW,
        #     "2008-09-01T00:00:00.000",
        #     "2008-09-01T00:05:00.000",
        #     "2008-09-01T00:05:00.000",
        # ),
        # (
        #     test_defs.Experiments.GAUSS3D,
        #     "2001-01-01T00:00:00.000",
        #     "2001-01-01T00:00:04.000",
        #     "2001-01-01T00:00:04.000",
        # ),
        (
            test_defs.Experiments.EXCLAIM_APE_AES,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
        ),
        # (
        #     test_defs.Experiments.MCH_CH_R04B09,
        #     "2021-06-20T12:00:00.000",
        #     "2021-06-20T12:00:10.000",
        #     "2021-06-20T12:00:10.000",
        # ),
        # (
        #     test_defs.Experiments.MCH_CH_R04B09,
        #     "2021-06-20T12:00:10.000",
        #     "2021-06-20T12:00:20.000",
        #     "2021-06-20T12:00:20.000",
        # ),
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
    (``time-step-exit``) savepoint. EXCLAIM_APE_AES additionally runs the physics
    (muphys + tmx, both auto-enabled by the config reader from ``aes_phy_nml`` /
    ``aes_vdf_nml``) and also validates the tracers. Exception: MCH_CH_R04B09 compares
    against the mid-time-step dynamics savepoints, because its reference runs NWP
    physics + limited-area nudging after the dynamics, which the driver does not (see
    the comment in the body). Per-field tolerances live in ``_TOLERANCES``.

    EXCLAIM_APE_AES, as of the **v08** reference, runs graupel AND vdf/tmx — vn and w
    are now actively written by the physics (tmx momentum coupling), unlike the
    muphys-only v06 era. Tolerances measured 2026-08-20 on v08 under the parallel
    two-layer coupling (see ``_TOLERANCES``); the residuals bundle the remaining
    zero-surface-flux seam, the parallel-vs-sequential splitting difference vs the
    ICON reference, and the clipping / vertical-extent items below — together they
    stay at the 1e-3-relative level (theta_v) or far below.

    muphys: runs ``MuphysScheme.AES_GRAUPEL`` -- the port of the exact ICON
    formulation that generated the reference. The tracer comparison carries
    residuals from gaps not yet ported:

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
    if experiment_description is test_defs.Experiments.EXCLAIM_APE_AES:
        # the production enablement path: the Fortran namelist reader itself switches
        # the physics on — muphys from aes_phy_nml, tmx from aes_vdf_nml
        assert config.muphys is not None, "muphys must be auto-enabled for APE_aes"
        assert config.tmx is not None, "tmx must be auto-enabled from aes_vdf_nml for APE_aes"
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



# @pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, experiment_description",
    [
        ("warm_bubble", test_defs.Experiments.WEISMAN_KLEMP_TORUS),
    ],
)
def test_warm_bubble(
    *,
    experiment_case: str,
    experiment_description: test_defs.ExperimentDescription,
    tmp_path: pathlib.Path,
    generate_torus_grid: Callable[..., pathlib.Path],
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)
    grid_path = pathlib.Path("/capstor/store/cscs/userlab/cwd01/cong/grids/Torus_Triangles_100km_x_100km_res500m.nc")
    # grid_path = generate_torus_grid(
    #     n_rows=116,
    #     n_cols=100,
    #     edge_length=100.0,
    # )

    driver_utils.configure_logging(
        logging_level="debug",
        print_distributed_debug_msg=False,
        process_props=process_props,
    )

    my_config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    experiment_config = config_io.read_yaml_str(
        my_config_path.read_text(), driver_config.ExperimentConfig
    )
    # config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    
    # test_experiment_config = driver_config.read_experiment_config_from_fortran(config_file_path)
    # breakpoint()
    grid_managers = driver_utils.create_grid_manager(
        grid_file_path=grid_path,
        vertical_grid_config=experiment_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
 
    ds, icon4py_driver = driver.run_driver(
        config=experiment_config,
        grid_manager=grid_managers,
        process_props=process_props,
        backend=backend,
    )

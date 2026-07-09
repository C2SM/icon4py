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

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx as tmx_module
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


def test_tmx_plumbing() -> None:
    """Config-layer TMX wiring: ExperimentConfig.tmx exists, defaults to None, TmxConfig constructs."""
    fields = {f.name: f for f in dataclasses.fields(driver_config.ExperimentConfig)}
    assert "tmx" in fields, "ExperimentConfig must have a 'tmx' field"
    assert fields["tmx"].default is None, "ExperimentConfig.tmx must default to None"
    assert tmx_module.TmxConfig() is not None


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, timeloop_date_exit, step_date_exit",
    [
        (
            test_defs.Experiments.JW,
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
        ),
        (
            test_defs.Experiments.GAUSS3D,
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
        ),
        (
            test_defs.Experiments.EXCLAIM_APE_AES,
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
        ),
        # TODO (jcanton,msimberg) add MCH_CH_R04B09 Experiment here in https://github.com/C2SM/icon4py/pull/1281
    ],
)
def test_standalone_driver(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_exit: str,
    step_date_exit: str,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """End-to-end standalone-driver validation over one large time step.

    Runs on both dry experiments (JW, GAUSS3D) and the moist AES aquaplanet
    (EXCLAIM_APE_AES), branching on whether physics is enabled (``config.muphys``):

    - No physics: the final prognostic state comes purely from the dynamical core,
      validated against the mid-time-step dynamics savepoints at tight tolerances.
    - muphys enabled: validated against the end-of-time-step (``time-step-exit``)
      savepoint. The driver runs muphys with ``MuphysScheme.ICON_NWP`` (the
      MuphysConfig default) -- the port of the exact icon-nwp formulation that
      generated the reference -- and the ICON reference runs graupel only (everything
      else off), so the full prognostic state including the fields muphys writes
      (exner, tracers) can be compared.

    The muphys-written fields use provisional tolerances. The exner/theta_v coupling
    now mirrors ICON exactly, but residual deviations remain from other phy2dyn gaps
    not yet ported, so the tolerances stay loose until measured on the regenerated
    archive:

    - exner / theta_v: ``scatter_to_prognostic`` recomputes exner via the exact EOS
      from the updated virtual temperature and diagnoses theta_v = Tv/exner, mirroring
      ICON's phy2dyn coupling (mo_interface_iconam_aes.f90), so the exner/rho/theta_v
      trio stays EOS-consistent. The exner (atol=1e-6) and theta_v (atol=2.0) bounds are
      still the pre-port provisional values; with the coupling ported they should
      tighten substantially -- limited now only by the tracer gaps below feeding into
      the virtual temperature -- and must be re-measured on the archive.
    - negative tracers: ICON clips them before (iqneg_d2p) and after (iqneg_p2d)
      physics; the driver does not.
    - vertical extent: ICON runs graupel on jks_cloudy..nlev (zmaxcloudy); muphys runs
      the full column.

    The muphys granule itself is validated in isolation against the aes-graupel
    savepoints in test_muphys_datatest.py.
    """
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_config(config_file_path)
    if experiment_description == test_defs.Experiments.EXCLAIM_APE_AES:
        assert config.muphys is not None, "muphys must be enabled for the APE_aes experiment"

    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            "end_of_simulation": datetime.datetime.fromisoformat(timeloop_date_exit).replace(
                tzinfo=datetime.timezone.utc
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
        # Dry experiments (no physics): the final prognostic state is produced by the
        # dynamical core, so validate it against the mid-time-step dynamics savepoints.
        # istep=2 (corrector), substep=5 (last dyn substep) and linit=False are constant
        # for JW and GAUSS3D.
        nonhydro_exit = data_provider.from_savepoint_nonhydro_exit(
            istep=2, date=step_date_exit, substep=5
        )
        diffusion_exit = data_provider.from_savepoint_diffusion_exit(
            linit=False, date=step_date_exit
        )
        test_utils.assert_dallclose(
            prognostics.vn.asnumpy(), diffusion_exit.vn().asnumpy(), atol=6e-7
        )
        test_utils.assert_dallclose(
            prognostics.w.asnumpy(), diffusion_exit.w().asnumpy(), atol=8e-9
        )
        test_utils.assert_dallclose(
            prognostics.rho.asnumpy(), nonhydro_exit.rho_new().asnumpy(), atol=9e-10
        )
        test_utils.assert_dallclose(
            prognostics.exner.asnumpy(), diffusion_exit.exner().asnumpy(), atol=2e-10
        )
        test_utils.assert_dallclose(
            prognostics.theta_v.asnumpy(), diffusion_exit.theta_v().asnumpy(), atol=1e-7
        )
        return

    # Physics enabled (EXCLAIM_APE_AES): validate the full prognostic state against the
    # end-of-time-step savepoint (see the docstring for the reference setup and the
    # provisional-tolerance rationale).
    time_step_exit = data_provider.from_savepoint_time_step_exit(date=step_date_exit)

    # fields graupel does not touch: expect the dry-run tolerances
    test_utils.assert_dallclose(prognostics.vn.asnumpy(), time_step_exit.vn().asnumpy(), atol=6e-7)
    test_utils.assert_dallclose(prognostics.w.asnumpy(), time_step_exit.w().asnumpy(), atol=8e-9)
    test_utils.assert_dallclose(
        prognostics.rho.asnumpy(), time_step_exit.rho().asnumpy(), atol=9e-10
    )

    # fields muphys writes: provisional tolerances, to be measured on the regenerated
    # archive (ICON4PY_DALLCLOSE_PRINT_INSTEAD_OF_FAIL=true) and tightened once the
    # fidelity gaps noted in the docstring are closed
    test_utils.assert_dallclose(
        prognostics.exner.asnumpy(), time_step_exit.exner().asnumpy(), atol=1e-6
    )
    test_utils.assert_dallclose(
        prognostics.theta_v.asnumpy(), time_step_exit.theta_v().asnumpy(), atol=2.0
    )

    tracers = prognostics.tracer
    for name, field in (
        ("qv", tracers.qv),
        ("qc", tracers.qc),
        ("qr", tracers.qr),
        ("qs", tracers.qs),
        ("qi", tracers.qi),
        ("qg", tracers.qg),
    ):
        assert field is not None, f"tracer {name} must be allocated for the APE_aes experiment"
        test_utils.assert_dallclose(
            field.asnumpy(),
            getattr(time_step_exit, name)().asnumpy(),
            atol=1e-10,
        )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, timeloop_date_exit",
    [
        (
            test_defs.Experiments.EXCLAIM_APE_AES,
            "2008-09-01T00:05:00.000",
        ),
    ],
)
def test_standalone_driver_moist_physics_with_tmx(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_exit: str,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    """Smoke test: one large time step over EXCLAIM_APE_AES with muphys + TMX enabled.

    TMX is injected into the config with default parameters, which match the
    aquaplanet namelist values used in the EXCLAIM_APE_AES experiment.

    Config injection: ``TmxConfig()`` (defaults) is used rather than
    ``TmxConfig.from_fortran_dict(atm_dict)`` because the atm_dict is not
    surfaced by ``read_config`` and re-reading it here would duplicate logic.
    The defaults match the APE aquaplanet namelist for all parameters that
    affect this smoke test.

    Assertions:
    - The physics driver has exactly the two registered processes ["muphys", "tmx"].
    - All prognostic fields (vn, w, exner, theta_v, rho) and moisture tracers
      (qv, qc, qi) are finite after the step.

    No savepoint equality is asserted for vn/w because TMX writes them by design.
    """
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_config(config_file_path)
    assert config.muphys is not None, "muphys must be enabled for the APE_aes experiment"

    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            "end_of_simulation": datetime.datetime.fromisoformat(timeloop_date_exit).replace(
                tzinfo=datetime.timezone.utc
            ),
        }
    )

    # Inject TMX with default parameters (defaults match the aquaplanet namelist;
    # use TmxConfig() rather than from_fortran_dict because atm_dict is internal
    # to read_config and re-reading it here would duplicate the loading logic).
    config = dataclasses.replace(config, tmx=tmx_module.TmxConfig())

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    ds, icon4py_driver = standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    granules = icon4py_driver.granules
    prognostic = ds.prognostics.current
    tracers = prognostic.tracer

    assert granules.physics is not None
    assert [p.name for p in granules.physics._processes] == ["muphys", "tmx"]
    for name, field in (
        ("vn", prognostic.vn),
        ("w", prognostic.w),
        ("exner", prognostic.exner),
        ("theta_v", prognostic.theta_v),
        ("rho", prognostic.rho),
        ("qv", tracers.qv),
        ("qc", tracers.qc),
        ("qi", tracers.qi),
    ):
        arr = field.asnumpy()
        assert np.isfinite(arr).all(), f"{name} has non-finite entries after muphys+tmx step"

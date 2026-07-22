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
    """End-to-end standalone-driver validation over one time step.

    Runs on both dycore-only experiments (e.g., JW, GAUSS3D) as well as full physics experiments
    (e.g., EXCLAIM_APE_AES):

    - No physics: the final prognostic state comes purely from the dynamical core,
      validated against the mid-time-step dynamics savepoints at tight tolerances.
    - muphys enabled: validated against the end-of-time-step (``time-step-exit``)
      savepoint. The driver runs muphys with ``MuphysScheme.AES_GRAUPEL`` (the
      MuphysConfig default) -- the port of the exact icon-nwp formulation that
      generated the reference. Graupel is the only *physics* parameterization active in
      the reference, so the dynamics + coupling fields (vn/w/rho/exner/theta_v) compare
      tightly; the tracer comparison is only partial (see the transport gap below).

    The muphys-written fields are validated at tolerances measured on the v6 archive.
    The exner/theta_v coupling mirrors ICON's exact EOS, so those two are tight; the
    tracer comparison carries residuals from gaps not yet ported:

    - exner / theta_v: ``scatter_to_prognostic`` recomputes exner via the exact EOS
      from the updated virtual temperature and diagnoses theta_v = Tv/exner, mirroring
      ICON's phy2dyn coupling (mo_interface_iconam_aes.f90), so the exner/rho/theta_v
      trio stays EOS-consistent. Measured on v6: exner ~3e-9 (atol=1e-8) and theta_v
      ~7e-9 relative (rtol=3e-8) -- both essentially exact.
    - tracer transport (KNOWN MISMATCH): the ICON reference ran with tracer advection
      ON (``ltransport=.TRUE.``, ihadv_tracer=2 / ivadv_tracer=3 = MIURA / PPM), but the
      standalone driver DISABLES it for this experiment (see ``config.read_config``). The
      MIURA/PPM scheme itself is ported + tested; what's missing is the driver's advection
      *inputs* -- it never computes airmass (rho*ddqz) or supplies live mass fluxes
      (TODO(OngChia)), so advection divides tracer density by a zero airmass -> 0/0 -> NaN.
      So this validates muphys in isolation, NOT the transport+muphys the reference ran.
      Over one 300 s step the advective change is small, so qv still passes at atol=2e-6
      and qc/qr/qs/qi/qg match bit-for-bit (atol=1e-10) -- but this is a config mismatch,
      not a match. TODO (Yilu): revisit once the driver computes airmass + wires the mass fluxes.
    - negative tracers: ICON clips them before (iqneg_d2p) and after (iqneg_p2d)
      physics; the driver does not. Contributes to the qv residual (max ~6.6e-7,
      ~3e-4 relative, in the moist lower troposphere).
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
    test_utils.assert_dallclose(
        prognostics.vn.asnumpy(), time_step_exit.vn().asnumpy(), atol=6e-7, err_msg="vn"
    )
    test_utils.assert_dallclose(
        prognostics.w.asnumpy(), time_step_exit.w().asnumpy(), atol=1e-8, err_msg="w"
    )
    test_utils.assert_dallclose(
        prognostics.rho.asnumpy(), time_step_exit.rho().asnumpy(), atol=9e-10, err_msg="rho"
    )

    # fields muphys writes: tolerances measured on the v6 archive (see docstring). The
    # exner/theta_v coupling is validated to ~1e-8 relative -- essentially exact.
    test_utils.assert_dallclose(
        prognostics.exner.asnumpy(), time_step_exit.exner().asnumpy(), atol=1e-8, err_msg="exner"
    )
    test_utils.assert_dallclose(
        prognostics.theta_v.asnumpy(),
        time_step_exit.theta_v().asnumpy(),
        rtol=3e-8,
        err_msg="theta_v",
    )

    tracers = prognostics.tracer
    # qc/qr/qs/qi/qg match the reference bit-for-bit; qv carries a small residual
    # (max ~6.6e-7, ~3e-4 relative, in the moist lower troposphere). NOTE: the reference
    # ran with tracer transport ON but the driver disables advection (it can't yet
    # compute airmass / wire mass fluxes -- see the docstring "tracer transport" gap), so
    # qv also absorbs the 1-step advective change, not just neg-clipping / vertical-extent.
    # TODO (Yilu): revisit once the driver computes airmass + mass fluxes (then transport+muphys).
    tracer_atol = {"qv": 2e-6, "qc": 1e-10, "qr": 1e-10, "qs": 1e-10, "qi": 1e-10, "qg": 1e-10}
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
            atol=tracer_atol[name],
            err_msg=name,
        )

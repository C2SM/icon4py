# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import pytest

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import main
from icon4py.model.testing import definitions as test_defs, grid_utils, serialbox as sb, test_utils
from icon4py.model.testing.fixtures.datatest import backend_like

from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, istep_exit, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
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
    ],
)
def test_standalone_driver(
    experiment: test_defs.Experiment,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    backend_like: model_backends.BackendLike,
    tmp_path: pathlib.Path,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    substep_exit: int,
    savepoint_diffusion_exit: sb.IconDiffusionExitSavepoint,
) -> None:
    grid_file_path = grid_utils._download_grid_file(experiment.grid)
    output_path = tmp_path / "ci_driver_output"
    ds, _ = main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_like,
        output_path=output_path,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = savepoint_diffusion_exit.exner()
    theta_sp = savepoint_diffusion_exit.theta_v()
    vn_sp = savepoint_diffusion_exit.vn()
    w_sp = savepoint_diffusion_exit.w()
    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=6e-7,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=8e-9,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.exner.asnumpy(), exner_sp.asnumpy(), atol=5e-11
    )

    assert test_utils.dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=6e-8,
    )

    assert test_utils.dallclose(ds.prognostics.current.rho.asnumpy(), rho_sp.asnumpy(), atol=9e-10)

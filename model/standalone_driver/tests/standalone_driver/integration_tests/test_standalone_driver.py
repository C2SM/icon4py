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
from icon4py.model.testing import definitions, grid_utils, serialbox as sb, test_utils
from icon4py.model.testing.fixtures.datatest import backend, backend_like

from ..fixtures import *


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, substep_exit, prep_adv, dyn_timestep, step_date_init, step_date_exit, timeloop_diffusion_linit_exit",  # istep_exit, substep_exit"#, timeloop_date_init, timeloop_date_exit, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.JW,
            2,
            False,
            1,
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
            False,
        )
    ],
)
def test_standalone_driver(
    backend_like,
    backend,
    tmp_path: pathlib.Path,
    timeloop_diffusion_savepoint_exit: sb.IconDiffusionExitSavepoint,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    experiment: definitions.Experiments,
    substep_exit,
    prep_adv,
    dyn_timestep,
    step_date_init,
    step_date_exit,
):
    """
    Currently, this is a only test to check if the driver runs from a grid file without verifying the end result.
    TODO(anyone): Modify this test for scientific validation after IO is ready.
    """

    backend_name = None
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k

    grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)

    output_path = tmp_path / f"ci_driver_output_for_backend_{backend_name}"
    ds = main.main(
        configuration_file_path="./",
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path=str(output_path),
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = timeloop_diffusion_savepoint_exit.exner()
    theta_sp = timeloop_diffusion_savepoint_exit.theta_v()
    vn_sp = timeloop_diffusion_savepoint_exit.vn()  # savepoint_nonhydro_exit.vn_new()
    w_sp = timeloop_diffusion_savepoint_exit.w()

    assert test_utils.dallclose(
        ds.prognostics.next.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=6e-12,
    )

    assert test_utils.dallclose(
        ds.prognostics.next.w.asnumpy(),
        w_sp.asnumpy(),
        atol=8e-14,
    )

    assert test_utils.dallclose(
        ds.prognostics.next.exner.asnumpy(),
        exner_sp.asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.next.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=4e-12,
    )

    assert test_utils.dallclose(ds.prognostics.next.rho.asnumpy(), rho_sp.asnumpy())

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

from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, substep_exit, step_date_exit, timeloop_diffusion_linit_exit",
    [(definitions.Experiments.JW, 2, "2008-09-01T00:05:00.000", False)],
)
def test_standalone_driver(
    backend_like: model_backends.BackendLike,
    backend: model_backends.BackendLike,
    tmp_path: pathlib.Path,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    experiment: definitions.Experiments,
    substep_exit: int,
    step_date_exit: str,
    timeloop_diffusion_savepoint_exit: sb.IconDiffusionExitSavepoint,
) -> None:
    """
    TODO(anyone): Modify this test for scientific validation after IO is ready.
    """

    backend_name = "embedded"
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k

    grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)

    output_path = tmp_path / f"ci_driver_output_for_backend_{backend_name}"
    ds = main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path=output_path,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = timeloop_diffusion_savepoint_exit.exner()  # savepoint_nonhydro_exit.exner_new() #
    theta_sp = (
        timeloop_diffusion_savepoint_exit.theta_v()
    )  # savepoint_nonhydro_exit.theta_v_new() #
    vn_sp = timeloop_diffusion_savepoint_exit.vn()  # savepoint_nonhydro_exit.vn_new() #
    w_sp = timeloop_diffusion_savepoint_exit.w()  # savepoint_nonhydro_exit.w_new() #

    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=1e-4,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=1e-4,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.exner.asnumpy(), exner_sp.asnumpy(), atol=1e-6
    )

    assert test_utils.dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=1e-4,
    )

    assert test_utils.dallclose(ds.prognostics.current.rho.asnumpy(), rho_sp.asnumpy(), atol=1e-5)

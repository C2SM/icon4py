# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import pytest

from icon4py.model.common import model_backends, model_options
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_utils, main
from icon4py.model.testing import definitions, grid_utils, serialbox as sb, test_utils
from icon4py.model.testing.fixtures.datatest import backend, backend_like

from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, istep_exit, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
    [
        (
            definitions.Experiments.JW,
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
    experiment: definitions.Experiments,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    backend_like: model_backends.BackendLike,
    tmp_path: pathlib.Path,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    substep_exit: int,
    timeloop_diffusion_savepoint_exit_standalone: sb.IconDiffusionExitSavepoint,
) -> None:
    backend_name = next(
        (k for k, v in model_backends.BACKENDS.items() if backend_like == v), "embedded"
    )
    grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)
    backend = model_options.customize_backend(
        program=None, backend=driver_utils.get_backend_from_name(backend_name)
    )
    if backend is not None and "dace_gpu" in backend.name:
        pytest.skip("dace_gpu backend time limit exceeds 45 minutes")
    array_ns = data_alloc.import_array_ns(backend)
    output_path = tmp_path / f"ci_driver_output_for_backend_{backend_name}"
    ds = main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path=output_path,
        array_ns=array_ns,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = timeloop_diffusion_savepoint_exit_standalone.exner()
    theta_sp = timeloop_diffusion_savepoint_exit_standalone.theta_v()
    vn_sp = timeloop_diffusion_savepoint_exit_standalone.vn()
    w_sp = timeloop_diffusion_savepoint_exit_standalone.w()
    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=9e-7,
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

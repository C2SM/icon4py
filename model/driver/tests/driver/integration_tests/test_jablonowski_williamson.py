# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import jablonowski_williamson as jabw
from icon4py.model.testing import datatest_utils as dt_utils, test_utils
from icon4py.model.testing.fixtures.datatest import backend

from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, rank",
    [
        (dt_utils.JABW_EXPERIMENT, 0),
    ],
)
def test_jabw_initial_condition(
    experiment,
    ranked_data_path,
    backend,
    rank,
    data_provider,
    grid_savepoint,
    icon_grid,
):
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    default_surface_pressure = data_alloc.constant_field(icon_grid, 1e5, dims.CellDim)

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = jabw.model_initialization_jabw(
        icon_grid,
        cell_geometry,
        edge_geometry,
        ranked_data_path.joinpath(f"{experiment}/ser_data"),
        backend,
        rank,
    )

    # note that w is not verified because we decided to force w to zero in python framework after discussion
    jabw_exit_savepoint = data_provider.from_savepoint_jabw_exit()
    assert test_utils.dallclose(
        prognostic_state_now.rho.asnumpy(),
        jabw_exit_savepoint.rho().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.exner.asnumpy(),
        jabw_exit_savepoint.exner().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.theta_v.asnumpy(),
        jabw_exit_savepoint.theta_v().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.vn.asnumpy(),
        jabw_exit_savepoint.vn().asnumpy(),
    )

    assert test_utils.dallclose(
        diagnostic_state.pressure.asnumpy(),
        jabw_exit_savepoint.pressure().asnumpy(),
    )

    assert test_utils.dallclose(
        diagnostic_state.temperature.asnumpy(),
        jabw_exit_savepoint.temperature().asnumpy(),
    )

    assert test_utils.dallclose(
        diagnostic_state.surface_pressure.asnumpy(),
        default_surface_pressure.asnumpy(),
    )

    assert test_utils.dallclose(
        solve_nonhydro_diagnostic_state.perturbed_exner_at_cells_on_model_levels.asnumpy(),
        data_provider.from_savepoint_diagnostics_initial().exner_pr().asnumpy(),
        atol=1.0e-14,
    )

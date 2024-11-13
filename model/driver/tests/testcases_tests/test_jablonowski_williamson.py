# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers
from icon4py.model.driver.testcases import jablonowski_williamson as jabw


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
    rank,
    data_provider,
    grid_savepoint,
    icon_grid,
):
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()

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
        rank,
    )

    # note that w is not verified because we decided to force w to zero in python framework after discussion
    assert helpers.dallclose(
        prognostic_state_now.rho.asnumpy(),
        data_provider.from_savepoint_jabw_final().rho().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_now.exner.asnumpy(),
        data_provider.from_savepoint_jabw_final().exner().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_now.theta_v.asnumpy(),
        data_provider.from_savepoint_jabw_final().theta_v().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_now.vn.asnumpy(),
        data_provider.from_savepoint_jabw_final().vn().asnumpy(),
    )

    assert helpers.dallclose(
        diagnostic_state.pressure.asnumpy(),
        data_provider.from_savepoint_jabw_final().pressure().asnumpy(),
    )

    assert helpers.dallclose(
        diagnostic_state.temperature.asnumpy(),
        data_provider.from_savepoint_jabw_final().temperature().asnumpy(),
    )

    assert helpers.dallclose(
        diagnostic_state.surface_pressure.asnumpy(),
        data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
    )

    assert helpers.dallclose(
        solve_nonhydro_diagnostic_state.exner_pr.asnumpy(),
        data_provider.from_savepoint_jabw_diagnostic().exner_pr().asnumpy(),
        atol=1.0e-14,
    )

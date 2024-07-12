# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from icon4py.model.common.test_utils import helpers, datatest_utils as dt_utils
from icon4py.model.driver import initialization_utils as init_utils


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
    ) = init_utils.model_initialization_jabw(
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
        diagnostic_state.pressure_sfc.asnumpy(),
        data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
    )

    assert helpers.dallclose(
        solve_nonhydro_diagnostic_state.exner_pr.asnumpy(),
        data_provider.from_savepoint_jabw_diagnostic().exner_pr().asnumpy(),
        atol=1.0e-14,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, rank",
    [
        (dt_utils.GAUSS3D_EXPERIMENT, 0),
    ],
)
def test_gauss3d_initial_condition(
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
    ) = init_utils.model_initialization_gauss3d(
        icon_grid,
        cell_geometry,
        edge_geometry,
        ranked_data_path.joinpath(f"{experiment}/ser_data"),
        rank,
    )

    # only verifying those assigned in the IC rather than all (at least for now)
    assert helpers.dallclose(
        prognostic_state_now.rho.asnumpy(),
        data_provider.from_savepoint_nonhydro_init(1, "2001-01-01T00:00:04.000", 0)
        .rho_now()
        .asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_now.exner.asnumpy(),
        data_provider.from_savepoint_nonhydro_init(1, "2001-01-01T00:00:04.000", 0)
        .exner_now()
        .asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_now.theta_v.asnumpy(),
        data_provider.from_savepoint_nonhydro_init(1, "2001-01-01T00:00:04.000", 0)
        .theta_v_now()
        .asnumpy(),
    )

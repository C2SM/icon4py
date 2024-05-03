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

from icon4py.model.common.test_utils.datatest_utils import JABW_EXPERIMENT
from icon4py.model.common.test_utils.helpers import (
    dallclose,
)
from icon4py.model.driver.initialization_utils import model_initialization_jabw


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, rank",
    [
        (JABW_EXPERIMENT, 0),
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
    ) = model_initialization_jabw(
        icon_grid,
        cell_geometry,
        edge_geometry,
        ranked_data_path.joinpath(f"{experiment}/ser_data"),
        rank,
    )

    # note that w is not verified because we decided to force w to zero in python framework after discussion
    assert dallclose(
        prognostic_state_now.rho.asnumpy(),
        data_provider.from_savepoint_jabw_final().rho().asnumpy(),
    )

    assert dallclose(
        prognostic_state_now.exner.asnumpy(),
        data_provider.from_savepoint_jabw_final().exner().asnumpy(),
    )

    assert dallclose(
        prognostic_state_now.theta_v.asnumpy(),
        data_provider.from_savepoint_jabw_final().theta_v().asnumpy(),
    )

    assert dallclose(
        prognostic_state_now.vn.asnumpy(),
        data_provider.from_savepoint_jabw_final().vn().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.pressure.asnumpy(),
        data_provider.from_savepoint_jabw_final().pressure().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.temperature.asnumpy(),
        data_provider.from_savepoint_jabw_final().temperature().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.pressure_sfc.asnumpy(),
        data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
    )

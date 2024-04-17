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

from icon4py.model.common.test_utils.helpers import (
    dallclose,
)
from icon4py.model.common.test_utils.datatest_utils import JABW_EXPERIMENT
from icon4py.model.driver.initialization_utils import model_initialization_jabw


@pytest.mark.datatest
@pytest.mark.skip
@pytest.mark.parametrize(
    "experiment, rank",
    [
        (JABW_EXPERIMENT, 0),
    ],
)
def test_jabw_initial_condition(
    experiment,
    datapath,
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
        icon_grid, cell_geometry, edge_geometry, datapath, rank
    )

    # note that w is not verified because we decided to force w to zero in python framework after discussion
    assert dallclose(
        data_provider.from_savepoint_jabw_final().rho().asnumpy(),
        prognostic_state_now.rho.asnumpy(),
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().exner().asnumpy(),
        prognostic_state_now.exner.asnumpy(),
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().theta_v().asnumpy(),
        prognostic_state_now.theta_v.asnumpy(),
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().vn().asnumpy(), prognostic_state_now.vn.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().pressure().asnumpy(),
        diagnostic_state.pressure.asnumpy(),
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().temperature().asnumpy(),
        diagnostic_state.temperature.asnumpy(),
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
        diagnostic_state.pressure_sfc.asnumpy(),
    )

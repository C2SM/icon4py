# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.driver.test_cases import gauss3d
from icon4py.model.testing import datatest_utils as dt_utils, helpers


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
    backend,
    rank,
    data_provider,
    grid_savepoint,
    icon_grid,
):
    edge_geometry = grid_savepoint.construct_edge_geometry()

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = gauss3d.model_initialization_gauss3d(
        icon_grid,
        edge_geometry,
        ranked_data_path.joinpath(f"{experiment}/ser_data"),
        backend,
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

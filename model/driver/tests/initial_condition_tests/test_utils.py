# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from icon4py.model.testing import helpers
from icon4py.model.driver.test_cases import utils


def test_hydrostatic_adjustment_ndarray():
    # TODO (Jacopo / Chia Rui) these tests could be better
    num_cells = 10
    num_levels = 10

    # Generate test data
    rho0 = 1.25
    exner0 = 0.935
    theta_v0 = 293.14
    wgtfac_c = 1.05 * np.ones((num_cells, num_levels))
    ddqz_z_half = np.ones((num_cells, num_levels))
    exner_ref_mc = 0.89 * np.ones((num_cells, num_levels))
    d_exner_dz_ref_ic = 0.0 * np.ones((num_cells, num_levels))
    theta_ref_mc = 312 * np.ones((num_cells, num_levels))
    theta_ref_ic = 312 * np.ones((num_cells, num_levels))
    rho = rho0 * np.ones((num_cells, num_levels))
    exner = exner0 * np.ones((num_cells, num_levels))
    theta_v = theta_v0 * np.ones((num_cells, num_levels))

    # Call the function
    r_rho, r_exner, r_theta_v = utils.hydrostatic_adjustment_ndarray(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho,
        exner,
        theta_v,
        num_levels,
    )

    assert r_rho.shape == (num_cells, num_levels)
    assert r_exner.shape == (num_cells, num_levels)
    assert r_theta_v.shape == (num_cells, num_levels)

    assert helpers.dallclose(
        r_rho[:, -1],
        rho0 * np.ones(num_cells),
    )
    assert helpers.dallclose(
        r_rho[:, :-1],
        1.0046424441749071 * np.ones((num_cells, num_levels - 1)),
    )
    assert helpers.dallclose(
        r_exner,
        exner0 * np.ones((num_cells, num_levels)),
    )
    assert helpers.dallclose(
        r_theta_v,
        theta_v0 * np.ones((num_cells, num_levels)),
    )


def test_hydrostatic_adjustment_constant_thetav_ndarray():
    # TODO (Jacopo / Chia Rui) these tests could be better
    num_cells = 10
    num_levels = 10

    # Generate test data
    rho0 = 1.25
    exner0 = 0.935
    theta_v0 = 293.14
    wgtfac_c = 1.05 * np.ones((num_cells, num_levels))
    ddqz_z_half = np.ones((num_cells, num_levels))
    exner_ref_mc = 0.89 * np.ones((num_cells, num_levels))
    d_exner_dz_ref_ic = 0.0 * np.ones((num_cells, num_levels))
    theta_ref_mc = 312 * np.ones((num_cells, num_levels))
    theta_ref_ic = 312 * np.ones((num_cells, num_levels))
    rho = rho0 * np.ones((num_cells, num_levels))
    exner = exner0 * np.ones((num_cells, num_levels))
    theta_v = theta_v0 * np.ones((num_cells, num_levels))

    # Call the function
    r_rho, r_exner = utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho,
        exner,
        theta_v,
        num_levels,
    )

    assert r_rho.shape == (num_cells, num_levels)
    assert r_exner.shape == (num_cells, num_levels)

    assert helpers.dallclose(
        r_rho,
        1.0046424441749071 * np.ones((num_cells, num_levels)),
    )
    assert helpers.dallclose(
        r_exner,
        exner0 * np.ones((num_cells, num_levels)),
    )

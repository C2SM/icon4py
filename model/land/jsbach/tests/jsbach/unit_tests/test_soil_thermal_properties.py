# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.land.jsbach.soil_thermal_properties import (
    fao_soil_thermal_properties,
    soil_thermal_grid,
)


def test_soil_thermal_grid_canonical_5_layer():
    # Canonical 5-layer JSBACH soil energy profile, dz = (0.065, 0.254, 0.913, 2.902,
    # 5.700) m (mo_sse_config_class.f90:293). soillev = cumulative layer bottoms.
    soillev = np.array([0.065, 0.319, 1.232, 4.134, 9.834])

    grid = soil_thermal_grid(soillev)

    np.testing.assert_allclose(grid.dz.asnumpy(), [0.065, 0.254, 0.913, 2.902, 5.700])
    np.testing.assert_allclose(grid.bots.asnumpy(), soillev)
    np.testing.assert_allclose(grid.mids.asnumpy(), [0.0325, 0.192, 0.7755, 2.683, 6.984])
    # zd1(k) = 1 / (mids(k+1) - mids(k)); the bottom entry is unused -> 0.
    mids = grid.mids.asnumpy()
    expected_zd1 = np.zeros(5)
    expected_zd1[:-1] = 1.0 / np.diff(mids)
    np.testing.assert_allclose(grid.zd1.asnumpy(), expected_zd1)


def test_fao_soil_thermal_properties():
    # FAO soil-type index -> volumetric heat capacity and conductivity
    # (mo_sse_init.f90:44-45 tables; heat_cond = vol_heat_cap * thermal_diffusivity).
    fao_index = np.array([0.0, 1.0, 5.0])
    num_levels = 4

    vol_heat_cap, heat_cond = fao_soil_thermal_properties(fao_index, num_levels)

    vhc = vol_heat_cap.asnumpy()
    hc = heat_cond.asnumpy()
    assert vhc.shape == (3, num_levels)
    # broadcast: every layer equals the per-cell value
    np.testing.assert_allclose(vhc[:, 0], [2.25e6, 1.93e6, 2.48e6])
    np.testing.assert_array_equal(vhc, vhc[:, :1].repeat(num_levels, axis=1))
    np.testing.assert_allclose(hc[:, 0], [2.25e6 * 7.4e-7, 1.93e6 * 8.7e-7, 2.48e6 * 6.7e-7])

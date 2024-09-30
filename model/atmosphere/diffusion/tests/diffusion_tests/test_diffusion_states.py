# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import icon4py.model.common.dimension as dims
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.common.test_utils import helpers


@pytest.mark.datatest
def test_verify_geofac_n2s_field_manipulation(interpolation_savepoint, icon_grid):
    geofac_n2s = interpolation_savepoint.geofac_n2s().asnumpy()
    interpolation_state = interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    geofac_c = interpolation_state.geofac_n2s_c.asnumpy()
    geofac_nbh = interpolation_state.geofac_n2s_nbh.asnumpy()
    assert np.count_nonzero(geofac_nbh) > 0
    cec_table = icon_grid.get_offset_provider("C2CEC").table
    assert np.allclose(geofac_c, geofac_n2s[:, 0])
    assert geofac_nbh[cec_table].shape == geofac_n2s[:, 1:].shape
    assert np.allclose(geofac_nbh[cec_table], geofac_n2s[:, 1:])

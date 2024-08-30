# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_advective_normal_wind_tendency_numpy(
    grid,
    z_kin_hor_e: np.array,
    coeff_gradekin: np.array,
    z_ekinh: np.array,
    zeta: np.array,
    vt: np.array,
    f_e: np.array,
    c_lin_e: np.array,
    z_w_con_c_full: np.array,
    vn_ie: np.array,
    ddqz_z_full_e: np.array,
) -> np.array:
    e2c = grid.connectivities[dims.E2CDim]
    z_ekinh_e2c = z_ekinh[e2c]
    coeff_gradekin = coeff_gradekin.reshape(e2c.shape)
    coeff_gradekin = np.expand_dims(coeff_gradekin, axis=-1)
    f_e = np.expand_dims(f_e, axis=-1)
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)

    ddt_vn_apc = -(
        (coeff_gradekin[:, 0] - coeff_gradekin[:, 1]) * z_kin_hor_e
        + (-coeff_gradekin[:, 0] * z_ekinh_e2c[:, 0] + coeff_gradekin[:, 1] * z_ekinh_e2c[:, 1])
        + vt * (f_e + 0.5 * np.sum(zeta[grid.connectivities[dims.E2VDim]], axis=1))
        + np.sum(z_w_con_c_full[e2c] * c_lin_e, axis=1)
        * (vn_ie[:, :-1] - vn_ie[:, 1:])
        / ddqz_z_full_e
    )
    return ddt_vn_apc


class TestComputeAdvectiveNormalWindTendency(StencilTest):
    PROGRAM = compute_advective_normal_wind_tendency
    OUTPUTS = ("ddt_vn_apc",)

    @staticmethod
    def reference(
        grid,
        z_kin_hor_e: np.array,
        coeff_gradekin: np.array,
        z_ekinh: np.array,
        zeta: np.array,
        vt: np.array,
        f_e: np.array,
        c_lin_e: np.array,
        z_w_con_c_full: np.array,
        vn_ie: np.array,
        ddqz_z_full_e: np.array,
        **kwargs,
    ) -> dict:
        ddt_vn_apc = compute_advective_normal_wind_tendency_numpy(
            grid,
            z_kin_hor_e,
            coeff_gradekin,
            z_ekinh,
            zeta,
            vt,
            f_e,
            c_lin_e,
            z_w_con_c_full,
            vn_ie,
            ddqz_z_full_e,
        )
        return dict(ddt_vn_apc=ddt_vn_apc)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[dims.E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        z_kin_hor_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        coeff_gradekin = random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=vpfloat)
        coeff_gradekin_new = as_1D_sparse_field(coeff_gradekin, dims.ECDim)
        z_ekinh = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        zeta = random_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)
        vt = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        f_e = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        c_lin_e = random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=wpfloat)
        z_w_con_c_full = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        vn_ie = random_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat)
        ddqz_z_full_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        ddt_vn_apc = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=coeff_gradekin_new,
            z_ekinh=z_ekinh,
            zeta=zeta,
            vt=vt,
            f_e=f_e,
            c_lin_e=c_lin_e,
            vn_ie=vn_ie,
            z_w_con_c_full=z_w_con_c_full,
            ddqz_z_full_e=ddqz_z_full_e,
            ddt_vn_apc=ddt_vn_apc,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn_and_graddiv_vn_and_vt import (
    compute_avg_vn_and_graddiv_vn_and_vt,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def compute_avg_vn_and_graddiv_vn_and_vt_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    e_flx_avg: np.ndarray,
    vn: np.ndarray,
    geofac_grdiv: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
) -> tuple[np.ndarray, ...]:
    e2c2eO = connectivities[dims.E2C2EODim]
    e2c2e = connectivities[dims.E2C2EDim]
    e_flx_avg = np.expand_dims(e_flx_avg, axis=-1)
    z_vn_avg = np.sum(vn[e2c2eO] * e_flx_avg, axis=1)
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    z_graddiv_vn = np.sum(
        np.where((e2c2eO != -1)[:, :, np.newaxis], vn[e2c2eO] * geofac_grdiv, 0), axis=1
    )
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    vt = np.sum(np.where((e2c2e != -1)[:, :, np.newaxis], vn[e2c2e] * rbf_vec_coeff_e, 0), axis=1)
    return z_vn_avg, z_graddiv_vn, vt


@pytest.mark.embedded_remap_error
class TestComputeAvgVnAndGraddivVnAndVt(StencilTest):
    PROGRAM = compute_avg_vn_and_graddiv_vn_and_vt
    OUTPUTS = ("z_vn_avg", "z_graddiv_vn", "vt")

    @static_reference
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        e_flx_avg: np.ndarray,
        vn: np.ndarray,
        geofac_grdiv: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_vn_avg, z_graddiv_vn, vt = compute_avg_vn_and_graddiv_vn_and_vt_numpy(
            connectivities,
            e_flx_avg,
            vn,
            geofac_grdiv,
            rbf_vec_coeff_e,
        )
        return dict(z_vn_avg=z_vn_avg, z_graddiv_vn=z_graddiv_vn, vt=vt)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_flx_avg = self.data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        geofac_grdiv = self.data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        rbf_vec_coeff_e = self.data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim, dtype=wpfloat)
        vn = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_vn_avg = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_graddiv_vn = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vt = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            e_flx_avg=e_flx_avg,
            vn=vn,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

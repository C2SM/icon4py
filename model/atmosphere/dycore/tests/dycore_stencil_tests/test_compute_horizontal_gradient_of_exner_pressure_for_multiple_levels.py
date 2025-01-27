# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import (
    flatten_first_two_dims,
    random_field,
    zero_field,
)
from icon4py.model.testing.helpers import StencilTest


class TestComputeHorizontalGradientOfExnerPressureForMultipleLevels(StencilTest):
    PROGRAM = compute_horizontal_gradient_of_exner_pressure_for_multiple_levels
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        inv_dual_edge_length: np.ndarray,
        z_exner_ex_pr: np.ndarray,
        zdiff_gradp: np.ndarray,
        ikoffset: np.ndarray,
        z_dexner_dz_c_1: np.ndarray,
        z_dexner_dz_c_2: np.ndarray,
        **kwargs,
    ) -> dict:
        def _apply_index_field(shape, to_index, neighbor_table, offset_field):
            indexed = np.zeros(shape)
            for iprimary in range(shape[0]):
                for isparse in range(shape[1]):
                    for ik in range(shape[2]):
                        indexed[iprimary, isparse, ik] = to_index[
                            neighbor_table[iprimary, isparse],
                            ik + offset_field[iprimary, isparse, ik],
                        ]
            return indexed

        e2c = connectivities[dims.E2CDim]
        full_shape = e2c.shape + zdiff_gradp.shape[1:]
        zdiff_gradp = zdiff_gradp.reshape(full_shape)
        ikoffset = ikoffset.reshape(full_shape)
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

        z_exner_ex_pr_at_kidx = _apply_index_field(full_shape, z_exner_ex_pr, e2c, ikoffset)
        z_dexner_dz_c_1_at_kidx = _apply_index_field(full_shape, z_dexner_dz_c_1, e2c, ikoffset)
        z_dexner_dz_c_2_at_kidx = _apply_index_field(full_shape, z_dexner_dz_c_2, e2c, ikoffset)

        def at_neighbor(i):
            return z_exner_ex_pr_at_kidx[:, i, :] + zdiff_gradp[:, i, :] * (
                z_dexner_dz_c_1_at_kidx[:, i, :]
                + zdiff_gradp[:, i, :] * z_dexner_dz_c_2_at_kidx[:, i, :]
            )

        sum_expr = at_neighbor(1) - at_neighbor(0)

        z_gradh_exner = inv_dual_edge_length * sum_expr
        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[dims.E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_exner_ex_pr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        zdiff_gradp = random_field(grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=vpfloat)
        ikoffset = zero_field(grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32)

        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(ikoffset.shape[0], ikoffset.shape[1]),
            )

        zdiff_gradp_new = flatten_first_two_dims(dims.ECDim, dims.KDim, field=zdiff_gradp)
        ikoffset_new = flatten_first_two_dims(dims.ECDim, dims.KDim, field=ikoffset)

        z_dexner_dz_c_1 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_dexner_dz_c_2 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_gradh_exner = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            zdiff_gradp=zdiff_gradp_new,
            ikoffset=ikoffset_new,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
            z_gradh_exner=z_gradh_exner,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.diffusion.stencils.temporary_fields_for_turbulence_diagnostics import (
    temporary_fields_for_turbulence_diagnostics,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestTemporaryFieldsForTurbulenceDiagnostics(StencilTest):
    PROGRAM = temporary_fields_for_turbulence_diagnostics
    OUTPUTS = ("div", "kh_c")

    @staticmethod
    def reference(
        grid,
        kh_smag_ec: np.array,
        vn: np.array,
        e_bln_c_s: np.array,
        geofac_div: np.array,
        diff_multfac_smag: np.array,
        **kwargs,
    ) -> dict:
        c2e = grid.connectivities[dims.C2EDim]
        c2ce = grid.get_offset_provider("C2CE").table

        geofac_div = np.expand_dims(geofac_div, axis=-1)
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        diff_multfac_smag = np.expand_dims(diff_multfac_smag, axis=0)

        vn_geofac = vn[c2e] * geofac_div[c2ce]
        div = np.sum(vn_geofac, axis=1)
        mul = kh_smag_ec[c2e] * e_bln_c_s[c2ce]
        summed = np.sum(mul, axis=1)
        kh_c = summed / diff_multfac_smag

        return dict(div=div, kh_c=kh_c)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        geofac_div = as_1D_sparse_field(
            random_field(grid, dims.CellDim, dims.C2EDim, dtype=wpfloat), dims.CEDim
        )
        kh_smag_ec = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        e_bln_c_s = as_1D_sparse_field(
            random_field(grid, dims.CellDim, dims.C2EDim, dtype=wpfloat), dims.CEDim
        )
        diff_multfac_smag = random_field(grid, dims.KDim, dtype=vpfloat)

        kh_c = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        div = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            kh_smag_ec=kh_smag_ec,
            vn=vn,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            diff_multfac_smag=diff_multfac_smag,
            kh_c=kh_c,
            div=div,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

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

import numpy as np
import pytest
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.metrics.metric_fields import (
    compute_ddqz_z_full,
    compute_ddqz_z_half,
    compute_vwind_expl_wgt,
    compute_z_mc,
)
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    dallclose,
    is_python,
    is_roundtrip,
    random_field,
    zero_field,
)


class TestComputeZMc(StencilTest):
    PROGRAM = compute_z_mc
    OUTPUTS = ("z_mc",)

    @staticmethod
    def reference(
        grid,
        z_ifc: np.array,
        **kwargs,
    ) -> dict:
        shp = z_ifc.shape
        z_mc = 0.5 * (z_ifc + np.roll(z_ifc, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(z_mc=z_mc)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        z_mc = zero_field(grid, CellDim, KDim)
        z_if = random_field(grid, CellDim, KDim, extend={KDim: 1})
        horizontal_start = int32(0)
        horizontal_end = grid.num_cells
        vertical_start = int32(0)
        vertical_end = grid.num_levels

        return dict(
            z_mc=z_mc,
            z_ifc=z_if,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )


def test_compute_ddq_z_half(icon_grid, metrics_savepoint, backend):
    if is_python(backend):
        pytest.skip("skipping: unsupported backend")
    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    z_ifc = metrics_savepoint.z_ifc()
    z_mc = zero_field(icon_grid, CellDim, KDim)
    nlevp1 = icon_grid.num_levels + 1
    k_index = as_field((KDim,), np.arange(nlevp1, dtype=int32))
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    ddq_z_half = zero_field(icon_grid, CellDim, KDim, extend={KDim: 1})

    compute_ddqz_z_half.with_backend(backend=backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        k=k_index,
        nlev=icon_grid.num_levels,
        ddqz_z_half=ddq_z_half,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlevp1,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(ddq_z_half.asnumpy(), ddq_z_half_ref.asnumpy())


def test_compute_ddqz_z_full(icon_grid, metrics_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    ddqz_z_full = zero_field(icon_grid, CellDim, KDim)
    inv_ddqz_z_full = zero_field(icon_grid, CellDim, KDim)

    compute_ddqz_z_full.with_backend(backend)(
        z_ifc=z_ifc,
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())


# @pytest.mark.datatest
# def test_compute_vwind_impl_wgt(icon_grid, metrics_savepoint, backend):
#     if is_roundtrip(backend):
#         pytest.skip("skipping: slow backend")
#
#     z_ddxn_z_half_e
#     z_ddxt_z_half_e
#     dual_edge_length
#     vwind_offctr
#     vwind_impl_wgt_full = zero_field(icon_grid, CellDim, KDim)
#     vwind_impl_wgt_ref = metrics_savepoint.vwind_impl_wgt()
#
#     compute_vwind_impl_wgt.with_backend(backend)(
#         z_ddxn_z_half_e=z_ddxn_z_half_e,
#         z_ddxt_z_half_e=z_ddxt_z_half_e,
#         dual_edge_length=dual_edge_length,
#         vwind_impl_wgt=vwind_impl_wgt_full,
#         vwind_offctr=vwind_offctr,
#         horizontal_start=int32(0),
#         horizontal_end=icon_grid.num_cells,
#         offset_provider = {"C2E": icon_grid.get_offset_provider("C2E")},
#     )
#
#     assert dallclose(vwind_impl_wgt_full.asnumpy(), vwind_impl_wgt_ref.asnumpy())


@pytest.mark.datatest
def test_compute_vwind_expl_wgt(icon_grid, metrics_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")

    vwind_expl_wgt_full = zero_field(icon_grid, CellDim)
    vwind_expl_wgt_ref = metrics_savepoint.vwind_expl_wgt()
    vwind_impl_wgt = metrics_savepoint.vwind_impl_wgt()

    compute_vwind_expl_wgt.with_backend(backend)(
        vwind_impl_wgt=vwind_impl_wgt,
        vwind_expl_wgt=vwind_expl_wgt_full,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    assert dallclose(vwind_expl_wgt_full.asnumpy(), vwind_expl_wgt_ref.asnumpy())

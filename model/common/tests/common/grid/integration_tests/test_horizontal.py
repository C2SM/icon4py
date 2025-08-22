# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from gt4py import next as gtx

from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.testing import datatest_utils as dt_utils
from .. import utils
from ...fixtures import *  # noqa: F401, F403


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_icon_start_end_index(experiment, dim, grid_savepoint):
    end_indices = grid_savepoint.end_index()
    start_indices = grid_savepoint.start_index()
    start_map, end_map = h_grid.get_start_end_idx_from_icon_arrays(dim, start_indices, end_indices)
    _assert_domain_map(start_map, start_indices[dim])
    _assert_domain_map(end_map, end_indices[dim])


def _assert_domain_map(index_map: dict[h_grid.Domain, gtx.int32], index_array: np.ndarray):
    same_index = False
    for d, index in index_map.items():
        if d.zone == h_grid.Zone.INTERIOR:
            same_index = index == index_array[h_grid._ICON_INTERIOR[d.dim]]
        if d.zone == h_grid.Zone.LOCAL:
            same_index = index == index_array[h_grid.ICON_LOCAL[d.dim]]
        if d.zone == h_grid.Zone.END:
            same_index = index == index_array[h_grid._ICON_END[d.dim]]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim]]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 1]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 2]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 3]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 4]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 5]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 6]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 7]
        if d.zone == h_grid.Zone.NUDGING:
            same_index = index == index_array[h_grid._ICON_NUDGING[d.dim]]
        if d.zone == h_grid.Zone.NUDGING_LEVEL_2:
            same_index = index == index_array[h_grid._ICON_NUDGING[d.dim] + 1]
        if d.zone == h_grid.Zone.HALO:
            same_index = index == index_array[h_grid._ICON_HALO[d.dim]]
        if d.zone == h_grid.Zone.HALO_LEVEL_2:
            same_index = index == index_array[h_grid._ICON_HALO[d.dim] - 1]
        if not same_index:
            raise AssertionError(f"Wrong index for {d.zone} zone in dimension {d.dim}: ")

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gt4py import next as gtx

from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.testing import definitions
from icon4py.model.testing.fixtures import experiment

from ...fixtures import *  # noqa: F403
from .. import utils


if TYPE_CHECKING:
    import numpy as np

    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_domain_bounds_start_index(
    dim: gtx.Dimension, grid_savepoint: sb.IconGridSavepoint
) -> None:
    grid_savepoint.start_index(dim)
    start_index_array = grid_savepoint.start_index(dim)
    _map_and_assert_array(dim, start_index_array)


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_domain_bounds_end_index(
    dim: gtx.Dimension, grid_savepoint: sb.IconGridSavepoint
) -> None:
    end_index_array = grid_savepoint.end_index(dim)
    _map_and_assert_array(dim, end_index_array)


def _map_and_assert_array(dim: gtx.Dimension, index_array: np.ndarray) -> None:  # noqa: PLR0912
    index_map = h_grid.map_icon_domain_bounds(dim, index_array)
    same_index = False
    for d, index in index_map.items():
        if d.zone == h_grid.Zone.INTERIOR:
            same_index = index == index_array[h_grid._INTERIOR[d.dim]]
        if d.zone == h_grid.Zone.LOCAL:
            same_index = index == index_array[h_grid._LOCAL[d.dim]]
        if d.zone == h_grid.Zone.END:
            same_index = index == index_array[h_grid._END[d.dim]]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim]]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 1]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 2]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 3]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 4]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 5]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 6]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8:
            same_index = index == index_array[h_grid._LATERAL_BOUNDARY[d.dim] + 7]
        if d.zone == h_grid.Zone.NUDGING:
            same_index = index == index_array[h_grid._NUDGING[d.dim]]
        if d.zone == h_grid.Zone.NUDGING_LEVEL_2:
            same_index = index == index_array[h_grid._NUDGING[d.dim] + 1]
        if d.zone == h_grid.Zone.HALO:
            same_index = index == index_array[h_grid._HALO[d.dim]]
        if d.zone == h_grid.Zone.HALO_LEVEL_2:
            same_index = index == index_array[h_grid._HALO[d.dim] - 1]
        if not same_index:
            raise AssertionError(f"Wrong index for {d.zone} zone in dimension {d.dim}: ")

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils.datatest_utils import (
    GRIDS_PATH,
    R02B04_GLOBAL,
    REGIONAL_EXPERIMENT,
)


r04b09_dsl_grid_path = GRIDS_PATH.joinpath(REGIONAL_EXPERIMENT)
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name

r02b04_global_grid_path = GRIDS_PATH.joinpath(R02B04_GLOBAL)
r02b04_global_data_file = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.tar.gz").name


def horizontal_dim():
    for dim in (dims.VertexDim, dims.EdgeDim, dims.CellDim):
        yield dim


def global_grid_domains(dim: dims.Dimension):
    zones = [
        h_grid.Zone.END,
        h_grid.Zone.LOCAL,
        h_grid.Zone.INTERIOR,
        h_grid.Zone.HALO,
        h_grid.Zone.HALO_LEVEL_2,
    ]

    yield from _domain(dim, zones)


def _domain(dim, zones):
    domain = h_grid.domain(dim)
    for zone in zones:
        try:
            yield domain(zone)
        except AssertionError:
            ...


def valid_boundary_zones_for_dim(dim: dims.Dimension):
    zones = [
        h_grid.Zone.LATERAL_BOUNDARY,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7,
        h_grid.Zone.NUDGING,
        h_grid.Zone.NUDGING_LEVEL_2,
    ]

    yield from _domain(dim, zones)

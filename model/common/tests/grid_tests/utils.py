# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Optional

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import grid_manager as gm, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils as gridtest_utils


r04b09_dsl_grid_path = dt_utils.GRIDS_PATH.joinpath(dt_utils.REGIONAL_EXPERIMENT)
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name

r02b04_global_grid_path = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL)
r02b04_global_data_file = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.tar.gz").name

R02B04_GLOBAL_NUM_CELLS = 20480
R02B04_GLOBAL_NUM_EDGES = 30720
R02B04_GLOBAL_NUM_VERTEX = 10242
managers = {}


def horizontal_dims():
    for d in vars(dims).values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.HORIZONTAL:
            yield d


def one_dimensional_sparse_dims():
    for d in vars(dims).values():
        if (
            isinstance(d, gtx.Dimension)
            and d.kind == gtx.DimensionKind.HORIZONTAL
            and d not in dims.MAIN_HORIZONTAL_DIMENSIONS.values()
        ):
            yield d


def main_horizontal_dims():
    yield from dims.MAIN_HORIZONTAL_DIMENSIONS.values()


def vertical_dims():
    for d in vars(dims).values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.VERTICAL:
            yield d


def non_horizontal_dims():
    yield from vertical_dims()
    yield from local_dims()


def local_dims():
    for d in vars(dims).values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.LOCAL:
            yield d


def horizontal_offsets():
    for d in vars(dims).values():
        if isinstance(d, gtx.FieldOffset) and len(d.target) == 2:
            yield d


def non_local_dims():
    yield from vertical_dims()
    yield from horizontal_dims()


def all_dims():
    yield from vertical_dims()
    yield from horizontal_dims()
    yield from local_dims()


def global_grid_domains(dim: gtx.Dimension):
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


def run_grid_manager(
    file: str, keep_skip_values: bool, backend: Optional[gtx_backend.Backend]
) -> gm.GridManager:
    key = "_".join(
        (file, data_alloc.backend_name(backend), "skip" if keep_skip_values else "no_skip")
    )
    if not managers.get(key):
        manager = gridtest_utils.get_grid_manager(
            file, keep_skip_values=keep_skip_values, num_levels=1, backend=backend
        )
        managers[key] = manager
    return managers.get(key)

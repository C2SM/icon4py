# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
import gt4py.next.typing as gtx_typing
import gt4py.next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import grid_manager as gm, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, grid_utils as gridtest_utils


managers: dict[str, gm.GridManager] = {}


def horizontal_dims() -> Iterator[gtx.Dimension]:
    for d in vars(dims).values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.HORIZONTAL:
            yield d


def main_horizontal_dims() -> Iterator[gtx.Dimension]:
    yield from dims.MAIN_HORIZONTAL_DIMENSIONS.values()


def vertical_dims() -> Iterator[gtx.Dimension]:
    for d in vars(dims).values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.VERTICAL:
            yield d


def non_horizontal_dims() -> Iterator[gtx.Dimension]:
    yield from vertical_dims()
    yield from local_dims()


def local_dims() -> Iterator[gtx.Dimension]:
    for d in vars(dims).values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.LOCAL:
            yield d


def horizontal_offsets() -> Iterator[gtx.FieldOffset]:
    for d in vars(dims).values():
        if isinstance(d, gtx.FieldOffset) and len(d.target) == 2:
            yield d


def non_local_dims() -> Iterator[gtx.Dimension]:
    yield from vertical_dims()
    yield from horizontal_dims()


def all_dims() -> Iterator[gtx.Dimension]:
    yield from vertical_dims()
    yield from horizontal_dims()
    yield from local_dims()


def global_grid_domains(dim: gtx.Dimension) -> Iterator[h_grid.Domain]:
    zones = [
        h_grid.Zone.END,
        h_grid.Zone.LOCAL,
        h_grid.Zone.INTERIOR,
        h_grid.Zone.HALO,
        h_grid.Zone.HALO_LEVEL_2,
    ]

    yield from _domain(dim, zones)


def _domain(dim: gtx.Dimension, zones: Sequence[h_grid.Zone]) -> Iterator[h_grid.Domain]:
    domain = h_grid.domain(dim)
    for zone in zones:
        with contextlib.suppress(AssertionError):
            yield domain(zone)


def valid_boundary_zones_for_dim(dim: gtx.Dimension) -> Iterator[h_grid.Domain]:
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
    grid: definitions.GridDescription,
    keep_skip_values: bool,
    backend: gtx_typing.Backend | None,
) -> gm.GridManager:
    key = "_".join(
        (grid.name, data_alloc.backend_name(backend), "skip" if keep_skip_values else "no_skip")
    )
    if (manager := managers.get(key)) is not None:
        return manager
    else:
        manager = gridtest_utils.get_grid_manager_from_identifier(
            grid,
            keep_skip_values=keep_skip_values,
            num_levels=1,
            backend=backend,
        )
        managers[key] = manager
        return manager

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.common import dimension as dims, model_backends
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


def _domain(dim: gtx.Dimension, zones: Iterator[h_grid.Zone]) -> Iterator[h_grid.Domain]:
    domain = h_grid.domain(dim)
    for zone in zones:
        with contextlib.suppress(AssertionError):
            yield domain(zone)


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
            allocator=model_backends.get_allocator(backend),
        )
        managers[key] = manager
        return manager


GRID_REFERENCE_VALUES = {
    definitions.Experiments.EXCLAIM_APE.name: {
        "mean_edge_length": 240221.1036647776,
        "mean_dual_edge_length": 138710.63736114913,
        "mean_cell_area": 24906292887.251026,
        "mean_dual_area": 49802858653.68937,
    },
    definitions.Experiments.MCH_CH_R04B09.name: {
        "mean_edge_length": 3803.019140934253,
        "mean_dual_edge_length": 2180.911493355989,
        "mean_cell_area": 6256048.940145881,
        "mean_dual_area": 12259814.063180268,
    },
    definitions.Experiments.GAUSS3D.name: {
        "mean_edge_length": 757.5757575757576,
        "mean_dual_edge_length": 437.3865675678984,
        "mean_cell_area": 248515.09520903317,
        "mean_dual_area": 497030.1904180664,
    },
}

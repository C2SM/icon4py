# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils.compute_nabla2_on_cell import compute_nabla2_on_cell
from icon4py.model.common.utils import data_allocation as data_alloc


@gtx.field_operator
def _update_smoothed_topography(
    smoothed_topography: fa.CellField[ta.wpfloat],
    nabla2_topo: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
) -> fa.CellField[ta.wpfloat]:
    """
    Updates the smoothed topography field inside the loop.
    """
    return smoothed_topography + 0.125 * nabla2_topo * cell_areas


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_smoothed_topography(
    nabla2_topo: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
    smoothed_topography: fa.CellField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _update_smoothed_topography(
        smoothed_topography=smoothed_topography,
        nabla2_topo=nabla2_topo,
        cell_areas=cell_areas,
        out=smoothed_topography,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )


def smooth_topography(
    topography: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    grid: base.BaseGrid,
    backend: gtx_backend.Backend,
    num_iterations: int = 25,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """

    smoothed_topography = gtx.as_field(
        (dims.CellDim,), topography.ndarray, allocator=backend
    )  # TODO (@halungge) this should copy?

    nabla2_topo = data_alloc.zero_field(grid, dims.CellDim, backend=backend)

    for _ in range(num_iterations):
        compute_nabla2_on_cell.with_backend(backend)(
            psi_c=smoothed_topography,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_topo,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            offset_provider={
                "C2E2CO": grid.get_connectivity("C2E2CO"),
            },
        )

        update_smoothed_topography.with_backend(backend)(
            nabla2_topo=nabla2_topo,
            cell_areas=cell_areas,
            smoothed_topography=smoothed_topography,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            offset_provider={},
        )

    return smoothed_topography

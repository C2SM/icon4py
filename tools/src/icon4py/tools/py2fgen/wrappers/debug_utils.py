# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common as gtx_common

from icon4py.model.common.decomposition import definitions
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.model.common.grid.icon import IconGrid
from icon4py.tools.common.logger import setup_logger


log = setup_logger(__name__)


def print_grid_decomp_info(
    icon_grid: IconGrid,
    processor_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,
    num_cells: int,
    num_edges: int,
    num_verts: int,
) -> None:
    log.info(
        "icon_grid:cell_start for rank %s is.... %s",
        processor_props.rank,
        icon_grid._start_indices[CellDim],
    )
    log.info(
        "icon_grid:cell_end for rank %s is.... %s",
        processor_props.rank,
        icon_grid._end_indices[CellDim],
    )
    log.info(
        "icon_grid:vert_start for rank %s is.... %s",
        processor_props.rank,
        icon_grid._start_indices[VertexDim],
    )
    log.info(
        "icon_grid:vert_end for rank %s is.... %s",
        processor_props.rank,
        icon_grid._end_indices[VertexDim],
    )
    log.info(
        "icon_grid:edge_start for rank %s is.... %s",
        processor_props.rank,
        icon_grid._start_indices[EdgeDim],
    )
    log.info(
        "icon_grid:edge_end for rank %s is.... %s",
        processor_props.rank,
        icon_grid._end_indices[EdgeDim],
    )

    for offset, connectivity in icon_grid.connectivities.items():
        if gtx_common.is_neighbor_table(connectivity):
            log.debug(
                f"icon_grid:{offset} for rank {processor_props.rank} is.... {connectivity.asnumpy()}"
            )

    log.info(
        "c_glb_index for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.global_index(CellDim)[0:num_cells],
    )
    log.info(
        "e_glb_index for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.global_index(EdgeDim)[0:num_edges],
    )
    log.info(
        "v_glb_index for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.global_index(VertexDim)[0:num_verts],
    )

    log.info(
        "c_owner_mask for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.owner_mask(CellDim)[0:num_cells],
    )
    log.info(
        "e_owner_mask for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.owner_mask(EdgeDim)[0:num_edges],
    )
    log.info(
        "v_owner_mask for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.owner_mask(VertexDim)[0:num_verts],
    )

    log.info(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels} "
        f"local cells = {decomposition_info.global_index(CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    log.info(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

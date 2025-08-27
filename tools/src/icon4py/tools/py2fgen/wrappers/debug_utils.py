# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import horizontal as h_grid, icon
from icon4py.tools.common import logger


log = logger.setup_logger(__name__)


def print_grid_decomp_info(
    icon_grid: icon.IconGrid,
    processor_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,
    num_cells: int,
    num_edges: int,
    num_verts: int,
) -> None:
    log.info(
        "icon_grid:cell_start for rank %s is.... %s",
        processor_props.rank,
        "\n".join(
            [
                f"{k:<10}  - {icon_grid.start_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.CellDim)
            ]
        ),
    )
    log.info(
        "icon_grid:cell_end for rank %s is.... %s",
        processor_props.rank,
        "\n".join(
            [
                f"{k:<10}  - {icon_grid.end_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.CellDim)
            ]
        ),
    )
    log.info(
        "icon_grid:vert_start for rank %s is.... %s",
        processor_props.rank,
        "\n".join(
            [
                f"{k:<10}  - {icon_grid.start_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.VertexDim)
            ]
        ),
    )
    log.info(
        "icon_grid:vert_end for rank %s is.... %s",
        processor_props.rank,
        "\n".join(
            [
                f"{k:<10}  - {icon_grid.end_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.VertexDim)
            ]
        ),
    )
    log.info(
        "icon_grid:edge_start for rank %s is.... %s",
        processor_props.rank,
        "\n".join(
            [
                f"{k:<10}  - {icon_grid.start_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.EdgeDim)
            ]
        ),
    )
    log.info(
        "icon_grid:edge_end for rank %s is.... %s",
        processor_props.rank,
        "\n".join(
            [
                f"{k:<10}  - {icon_grid.end_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.EdgeDim)
            ]
        ),
    )

    for offset, connectivity in icon_grid.connectivities.items():
        if gtx_common.is_neighbor_table(connectivity):
            log.debug(
                f"icon_grid:{offset} for rank {processor_props.rank} is.... {connectivity.asnumpy()}"
            )

    log.info(
        "c_glb_index for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.global_index(dims.CellDim)[0:num_cells],
    )
    log.info(
        "e_glb_index for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.global_index(dims.EdgeDim)[0:num_edges],
    )
    log.info(
        "v_glb_index for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.global_index(dims.VertexDim)[0:num_verts],
    )

    log.info(
        "c_owner_mask for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.owner_mask(dims.CellDim)[0:num_cells],
    )
    log.info(
        "e_owner_mask for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.owner_mask(dims.EdgeDim)[0:num_edges],
    )
    log.info(
        "v_owner_mask for rank %s is.... %s",
        processor_props.rank,
        decomposition_info.owner_mask(dims.VertexDim)[0:num_verts],
    )

    log.info(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels} "
        f"local cells = {decomposition_info.global_index(dims.CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    log.info(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

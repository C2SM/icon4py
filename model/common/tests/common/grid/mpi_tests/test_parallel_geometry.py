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

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry_attributes as attrs, geometry, gridfile
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb



@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, grid_name",
    [
        ("edge_area", "edge_areas"),
        ("edge_midpoint_to_cell_center_distance", "edge_cell_length"),
        ("eastward_component_of_edge_tangent_on_vertex", "dual_normal_vert_x"),
        ("edge_midpoint_to_vertex_distance", "edge_vert_length"),
        ("orientation_of_normal_to_cell_edges", "edge_orientation"),
        ("grid_latitude_of_vertex", "verts_vertex_lat"),
        ("eastward_component_of_edge_tangent_on_cell", "dual_normal_cell_x"),
        ("cell_area", "cell_areas"),
    ],
)
def test_distributed_geometry_attrs(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    attrs_name: str,
    grid_name: str,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    grid = grid_savepoint.construct_icon_grid(backend)
    coordinates = grid_savepoint.coordinates()
    extra_fields = {gridfile.GeometryName.CELL_AREA: grid_savepoint.cell_areas(), gridfile.GeometryName.EDGE_LENGTH:grid_savepoint.primal_edge_length(),
                    gridfile.GeometryName.DUAL_EDGE_LENGTH:grid_savepoint.dual_edge_length(),
                    gridfile.GeometryName.EDGE_CELL_DISTANCE: grid_savepoint.edge_cell_length(),
                    gridfile.GeometryName.EDGE_VERTEX_DISTANCE: grid_savepoint.edge_vert_length(),
                    gridfile.GeometryName.DUAL_AREA: grid_savepoint.vertex_dual_area(),
                    gridfile.GeometryName.TANGENT_ORIENTATION:grid_savepoint.tangent_orientation(),
                    gridfile.GeometryName.CELL_NORMAL_ORIENTATION: grid_savepoint.edge_orientation(),
                    gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX: grid_savepoint.vertex_edge_orientation()
                    }



    exchange = decomposition.create_exchange(processor_props, decomposition_info)

    #grid_geometry = grid_utils.get_grid_geometry(backend, experiment, exchange=exchange, decomposition_info=decomposition_info)
    grid_geometry = geometry.GridGeometry(grid = grid, decomposition_info=decomposition_info, backend=backend,
                                          metadata =attrs.attrs, coordinates=coordinates, extra_fields=extra_fields,
                                          exchange=exchange)
    field_ref = grid_savepoint.__getattribute__(grid_name)().asnumpy()
    field = grid_geometry.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, equal_nan=True, atol=1e-13)

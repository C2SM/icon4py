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
from icon4py.model.common.grid import geometry, geometry_attributes as attrs, gridfile
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    parallel_geometry_grid,
    processor_props,
    ranked_data_path,
)


if TYPE_CHECKING:
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
        ("northward_component_of_edge_tangent_on_cell", "dual_normal_cell_y"),
        ("cell_area", "cell_areas"),
    ],
)
def test_distributed_geometry_attrs(
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    parallel_geometry_grid: geometry.GridGeometry,
    attrs_name: str,
    grid_name: str,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    grid_geometry = parallel_geometry_grid
    field_ref = grid_savepoint.__getattribute__(grid_name)().asnumpy()
    field = grid_geometry.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, equal_nan=True, atol=1e-12)

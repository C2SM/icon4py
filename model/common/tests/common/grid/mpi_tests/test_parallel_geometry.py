# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry, geometry_attributes as attrs, horizontal as h_grid
from icon4py.model.common.math import helpers as math_helpers
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    geometry_from_savepoint,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)
from .. import utils


try:
    from icon4py.model.common.decomposition import mpi_decomposition
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

if TYPE_CHECKING:
    from icon4py.model.testing import serialbox as sb

edge_domain = h_grid.domain(dims.EdgeDim)
lb_local = edge_domain(h_grid.Zone.LOCAL)
lb_lateral = edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, grid_name",
    [
        (attrs.EDGE_AREA, "edge_areas"),
        (attrs.EDGE_NORMAL_U, "primal_normal_v1"),
        (attrs.EDGE_NORMAL_V, "primal_normal_v2"),
        (attrs.EDGE_NORMAL_VERTEX_U, "primal_normal_vert_x"),
        (attrs.EDGE_NORMAL_VERTEX_V, "primal_normal_vert_y"),
        (attrs.EDGE_NORMAL_CELL_U, "primal_normal_cell_x"),
        (attrs.EDGE_NORMAL_CELL_V, "primal_normal_cell_y"),
        (attrs.EDGE_TANGENT_CELL_U, "dual_normal_cell_x"),
        (attrs.EDGE_TANGENT_VERTEX_U, "dual_normal_vert_x"),
        (attrs.EDGE_TANGENT_VERTEX_V, "dual_normal_vert_y"),
    ],
)
def test_distributed_geometry_attrs(
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
    attrs_name: str,
    grid_name: str,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    grid_geometry = geometry_from_savepoint
    field_ref = grid_savepoint.__getattribute__(grid_name)().asnumpy()
    field = grid_geometry.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, atol=1e-12)


@pytest.mark.xfail(reason="Wrong results")
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, grid_name, lb_domain",
    (
        ("inverse_of_" + attrs.DUAL_EDGE_LENGTH, "inv_dual_edge_length", lb_lateral),
        ("inverse_of_" + attrs.VERTEX_VERTEX_LENGTH, "inv_vert_vert_length", lb_local),
        ("inverse_of_" + attrs.EDGE_LENGTH, "inverse_primal_edge_lengths", lb_local),
    ),
)
def test_distributed_geometry_attrs_for_inverse(
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
    attrs_name: str,
    grid_name: str,
    lb_domain: h_grid.Domain,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    grid_geometry = geometry_from_savepoint
    field_ref = grid_savepoint.__getattribute__(grid_name)().asnumpy()
    field = grid_geometry.get(attrs_name).asnumpy()
    lb = grid_geometry.grid.start_index(lb_domain)
    assert test_utils.dallclose(field[lb:], field_ref[lb:], rtol=5e-10)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, grid_name",
    [
        (attrs.CORIOLIS_PARAMETER, "f_e"),
        (attrs.EDGE_TANGENT_X, "dual_cart_normal_x"),
        (attrs.EDGE_TANGENT_Y, "dual_cart_normal_y"),
        (attrs.EDGE_TANGENT_Z, "dual_cart_normal_z"),
        (attrs.EDGE_NORMAL_X, "primal_cart_normal_x"),
        (attrs.EDGE_NORMAL_Y, "primal_cart_normal_y"),
        (attrs.EDGE_NORMAL_Z, "primal_cart_normal_z"),
    ],
)
def test_geometry_attr_no_halos(
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
    attrs_name: str,
    grid_name: str,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    grid_geometry = geometry_from_savepoint
    field_ref = grid_savepoint.__getattribute__(grid_name)().asnumpy()
    field = grid_geometry.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, equal_nan=True, atol=1e-12)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "x, y, z, dimension",
    [
        (attrs.CELL_CENTER_X, attrs.CELL_CENTER_Y, attrs.CELL_CENTER_Z, dims.CellDim),
        (attrs.EDGE_CENTER_X, attrs.EDGE_CENTER_Y, attrs.EDGE_CENTER_Z, dims.EdgeDim),
        (attrs.VERTEX_X, attrs.VERTEX_Y, attrs.VERTEX_Z, dims.VertexDim),
    ],
)
def test_cartesian_geometry_attr_no_halos(
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
    x: str,
    y: str,
    z: str,
    dimension: gtx.Dimension,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    grid_geometry = geometry_from_savepoint
    x_field = grid_geometry.get(x)
    y_field = grid_geometry.get(y)
    z_field = grid_geometry.get(z)
    norm = data_alloc.zero_field(
        grid_geometry.grid, dimension, dtype=x_field.dtype, allocator=backend
    )
    math_helpers.norm2_on_vertices(x_field, z_field, y_field, out=norm, offset_provider={})
    assert test_utils.dallclose(norm.asnumpy(), 1.0)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attr_name", ["mean_edge_length", "mean_dual_edge_length", "mean_cell_area", "mean_dual_area"]
)
def test_distributed_geometry_mean_fields(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
    attr_name: str,
) -> None:
    if processor_props.comm_size > 1:
        pytest.skip("Values not serialized for multiple processors")

    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    assert hasattr(experiment, "name")
    value_ref = utils.GRID_REFERENCE_VALUES[experiment.name][attr_name]
    value = geometry_from_savepoint.get(attr_name)
    assert value == pytest.approx(value_ref)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_distributed_mean_cell_area(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    value_ref = grid_savepoint.mean_cell_area()
    value = geometry_from_savepoint.get("mean_cell_area")
    assert value == pytest.approx(value_ref, rel=1e-1)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_distributed_mean_dual_edge_length(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    geometry_from_savepoint: geometry.GridGeometry,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)

    value_ref = np.mean(grid_savepoint.dual_edge_length().asnumpy())
    value = geometry_from_savepoint.get("mean_dual_edge_length")
    assert value == pytest.approx(value_ref, rel=1e-1)

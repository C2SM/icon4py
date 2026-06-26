# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Construct the icon4py static-field factories for the v2 bindings.

The factories derive the interpolation and metric fields that the v1 bindings receive
precomputed from ICON. Two field sets are instead injected from Fortran via
`PrecomputedFieldProvider`, overriding the factory computation:
- RBF vector coefficients: rounding-sensitive, must match ICON bit-compatibly enough.
- `mean_cell_area`: computed in icon4py with a global reduction that may differ from ICON.
"""

import dataclasses

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes,
    gridfile,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.grid.grid_manager import CoordinateDict, GeometryDict
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import factory as states_factory


def single_node_decomposition_info(
    *, num_cells: int, num_edges: int, num_vertices: int
) -> decomposition_defs.DecompositionInfo:
    """Trivial decomposition for a single rank: every entity is locally owned."""
    info = decomposition_defs.DecompositionInfo()
    for dim, n in (
        (dims.CellDim, num_cells),
        (dims.EdgeDim, num_edges),
        (dims.VertexDim, num_vertices),
    ):
        info.set_dimension(
            dim,
            np.arange(n, dtype=np.int32),
            np.ones(n, dtype=bool),
            None,
        )
    return info


@dataclasses.dataclass(frozen=True)
class StaticFieldSources:
    geometry: grid_geometry.GridGeometry
    interpolation: interpolation_factory.InterpolationFieldsFactory
    metrics: metrics_factory.MetricsFieldsFactory


def build_static_field_sources(
    *,
    grid: icon_grid.IconGrid,
    decomposition_info: decomposition_defs.DecompositionInfo,
    coordinates: CoordinateDict,
    extra_fields: GeometryDict,
    vertical_grid: v_grid.VerticalGrid,
    topography: fa.CellField[ta.wpfloat],
    interpolation_config: interpolation_factory.InterpolationConfig,
    metrics_config: metrics_factory.MetricsConfig,
    rbf_vec_coeff_v1: gtx.Field,
    rbf_vec_coeff_v2: gtx.Field,
    mean_cell_area: float,
    backend: gtx_typing.Backend | None,
    rbf_vec_coeff_e: gtx.Field | None = None,
    exchange: decomposition_defs.ExchangeRuntime = decomposition_defs.single_node_exchange,
    reductions: decomposition_defs.Reductions = decomposition_defs.single_node_reductions,
) -> StaticFieldSources:
    geometry = grid_geometry.GridGeometry(
        grid=grid,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=coordinates,
        extra_fields=extra_fields,
        metadata=geometry_attributes.attrs,
        exchange=exchange,
        global_reductions=reductions,
    )
    # Override icon4py's global-reduction mean_cell_area with ICON's value; this also
    # feeds characteristic_length = sqrt(mean_cell_area) and CellParams.
    geometry.register_provider(
        states_factory.PrecomputedFieldProvider(
            {geometry_attributes.MEAN_CELL_AREA: mean_cell_area}
        )
    )

    interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        config=interpolation_config,
        exchange=exchange,
    )
    rbf_fields = {
        interpolation_attributes.RBF_VEC_COEFF_V1: rbf_vec_coeff_v1,
        interpolation_attributes.RBF_VEC_COEFF_V2: rbf_vec_coeff_v2,
    }
    if rbf_vec_coeff_e is not None:
        rbf_fields[interpolation_attributes.RBF_VEC_COEFF_E] = rbf_vec_coeff_e
    interpolation.register_provider(states_factory.PrecomputedFieldProvider(rbf_fields))

    metrics = metrics_factory.MetricsFieldsFactory(
        grid=grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry,
        topography=topography,
        interpolation_source=interpolation,
        backend=backend,
        metadata=metrics_attributes.attrs,
        config=metrics_config,
        exchange=exchange,
        global_reductions=reductions,
    )

    return StaticFieldSources(geometry=geometry, interpolation=interpolation, metrics=metrics)


def build_extra_fields(
    *,
    edge_length: gtx.Field,
    dual_edge_length: gtx.Field,
    edge_cell_distance: gtx.Field,
    edge_vertex_distance: gtx.Field,
    cell_area: gtx.Field,
    dual_area: gtx.Field,
    tangent_orientation: gtx.Field,
    cell_normal_orientation: gtx.Field,
    edge_orientation_on_vertex: gtx.Field,
) -> GeometryDict:
    return {
        gridfile.GeometryName.EDGE_LENGTH: edge_length,
        gridfile.GeometryName.DUAL_EDGE_LENGTH: dual_edge_length,
        gridfile.GeometryName.EDGE_CELL_DISTANCE: edge_cell_distance,
        gridfile.GeometryName.EDGE_VERTEX_DISTANCE: edge_vertex_distance,
        gridfile.GeometryName.CELL_AREA: cell_area,
        gridfile.GeometryName.DUAL_AREA: dual_area,
        gridfile.GeometryName.TANGENT_ORIENTATION: tangent_orientation,
        gridfile.GeometryName.CELL_NORMAL_ORIENTATION: cell_normal_orientation,
        gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX: edge_orientation_on_vertex,
    }


def build_coordinates(
    *,
    cell_lat: gtx.Field,
    cell_lon: gtx.Field,
    edge_lat: gtx.Field,
    edge_lon: gtx.Field,
    vertex_lat: gtx.Field,
    vertex_lon: gtx.Field,
) -> CoordinateDict:
    return {
        dims.CellDim: {"lat": cell_lat, "lon": cell_lon},
        dims.EdgeDim: {"lat": edge_lat, "lon": edge_lon},
        dims.VertexDim: {"lat": vertex_lat, "lon": vertex_lon},
    }

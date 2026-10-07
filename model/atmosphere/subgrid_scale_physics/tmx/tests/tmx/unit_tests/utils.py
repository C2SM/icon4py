# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Random but physically plausible static states of tmx on the simple grid."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@dataclasses.dataclass(frozen=True)
class StaticStates:
    metric_state: tmx_states.TmxMetricState
    interpolation_state: tmx_states.TmxInterpolationState
    edge_params: grid_states.EdgeParams
    cell_params: grid_states.CellParams


def _extrapolation_coefficients(
    grid: base_grid.Grid,
    horizontal_dim: gtx.Dimension,
    k_start: int,
    allocator: gtx_typing.Allocator | None,
) -> gtx.Field:
    """
    Random stand-ins for the quadratic extrapolation coefficients `wgtfacq*` (to the surface,
    `k_start = nlev - 3`, or to the model top, `k_start = 0`): three rows, aligned to the
    levels they multiply.
    """
    size = grid.size[horizontal_dim]
    return gtx.as_field(
        gtx.domain({horizontal_dim: (0, size), dims.KDim: (k_start, k_start + 3)}),
        np.random.default_rng().uniform(0.3, 0.7, (size, 3)),
        allocator=allocator,
    )


def random_static_states(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> StaticStates:
    random_field = functools.partial(data_alloc.random_field, grid, allocator=allocator)
    positive = functools.partial(random_field, low=0.1, high=1.0)
    weight = functools.partial(random_field, low=0.3, high=0.7)

    metric_state = tmx_states.TmxMetricState(
        ddqz_z_full=positive(dims.CellDim, dims.KDim),
        inv_ddqz_z_full=positive(dims.CellDim, dims.KDim),
        ddqz_z_half=positive(dims.CellDim, dims.KHalfDim),
        inv_ddqz_z_half=positive(dims.CellDim, dims.KHalfDim),
        inv_ddqz_z_full_e=positive(dims.EdgeDim, dims.KDim),
        inv_ddqz_z_half_e=positive(dims.EdgeDim, dims.KHalfDim),
        inv_ddqz_z_half_v=positive(dims.VertexDim, dims.KHalfDim),
        wgtfac_c=weight(dims.CellDim, dims.KHalfDim),
        wgtfac_e=weight(dims.EdgeDim, dims.KHalfDim),
        wgtfacq_c=_extrapolation_coefficients(grid, dims.CellDim, grid.num_levels - 3, allocator),
        wgtfacq1_c=_extrapolation_coefficients(grid, dims.CellDim, 0, allocator),
        wgtfacq_e=_extrapolation_coefficients(grid, dims.EdgeDim, grid.num_levels - 3, allocator),
        wgtfacq1_e=_extrapolation_coefficients(grid, dims.EdgeDim, 0, allocator),
        geopot_agl_ifc=random_field(dims.CellDim, dims.KHalfDim, low=100.0, high=1.0e5),
        height_above_ground=random_field(dims.CellDim, dims.KDim, low=10.0, high=1.0e4),
    )
    interpolation_state = tmx_states.TmxInterpolationState(
        c_lin_e=weight(dims.EdgeDim, dims.E2CDim),
        e_bln_c_s=weight(dims.CellDim, dims.C2EDim),
        geofac_div=weight(dims.CellDim, dims.C2EDim),
        cells_aw_verts=weight(dims.VertexDim, dims.V2CDim),
        rbf_coeff_v1=weight(dims.VertexDim, dims.V2EDim),
        rbf_coeff_v2=weight(dims.VertexDim, dims.V2EDim),
        rbf_coeff_e=weight(dims.EdgeDim, dims.E2C2EDim),
        rbf_coeff_c1=weight(dims.CellDim, dims.C2E2C2EDim),
        rbf_coeff_c2=weight(dims.CellDim, dims.C2E2C2EDim),
    )
    unit = functools.partial(random_field, low=-1.0, high=1.0)
    inverse_length = functools.partial(random_field, dims.EdgeDim, low=1.0e-5, high=1.0e-3)
    edge_params = grid_states.EdgeParams(
        tangent_orientation=unit(dims.EdgeDim),
        inverse_primal_edge_lengths=inverse_length(),
        inverse_dual_edge_lengths=inverse_length(),
        inverse_vertex_vertex_lengths=inverse_length(),
        primal_normal_vert=(unit(dims.EdgeDim, dims.E2C2VDim), unit(dims.EdgeDim, dims.E2C2VDim)),
        dual_normal_vert=(unit(dims.EdgeDim, dims.E2C2VDim), unit(dims.EdgeDim, dims.E2C2VDim)),
        primal_normal_cell=(unit(dims.EdgeDim, dims.E2CDim), unit(dims.EdgeDim, dims.E2CDim)),
        dual_normal_cell=(unit(dims.EdgeDim, dims.E2CDim), unit(dims.EdgeDim, dims.E2CDim)),
        edge_areas=random_field(dims.EdgeDim, low=1.0e6, high=1.0e8),
        coriolis_frequency=unit(dims.EdgeDim),
        edge_center=(unit(dims.EdgeDim), unit(dims.EdgeDim)),
        primal_normal=(unit(dims.EdgeDim), unit(dims.EdgeDim)),
        edge_cell_distances=random_field(dims.EdgeDim, dims.E2CDim, low=1.0e3, high=1.0e4),
    )
    cell_params = grid_states.CellParams(
        cell_center_lat=unit(dims.CellDim),
        cell_center_lon=unit(dims.CellDim),
        area=random_field(dims.CellDim, low=1.0e6, high=1.0e8),
    )
    return StaticStates(
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )

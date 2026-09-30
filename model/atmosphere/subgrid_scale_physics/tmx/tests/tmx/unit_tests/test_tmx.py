# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test of the tmx component on the simple grid.

Runs one step on random but physically plausible fields and checks that every output is
finite. Correctness is covered by the stencil tests and the integration datatests.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config, tmx, tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


def _extrapolation_coefficients(
    grid: base_grid.Grid,
    horizontal_dim: gtx.Dimension,
    k_start: int,
    allocator: gtx_typing.Allocator | None,
) -> gtx.Field:
    """Three coefficient rows, aligned to the levels they multiply."""
    size = grid.size[horizontal_dim]
    return gtx.as_field(
        gtx.domain({horizontal_dim: (0, size), dims.KDim: (k_start, k_start + 3)}),
        np.random.default_rng().uniform(0.3, 0.7, (size, 3)),
        allocator=allocator,
    )


def test_tmx_run_smoke(backend_like: model_backends.BackendLike) -> None:
    allocator = model_backends.get_allocator(backend_like)
    grid = simple.simple_grid(allocator=allocator, num_levels=10)
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
    component = tmx.Tmx(
        grid=grid,
        config=tmx_config.TmxConfig(),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend_like,
        exchange=decomposition.SingleNodeExchange(),
    )

    wind = functools.partial(random_field, dims.CellDim, dims.KDim, low=-10.0, high=10.0)
    tracer = functools.partial(random_field, dims.CellDim, dims.KDim, low=0.0, high=1.0e-3)
    temperature = functools.partial(random_field, dims.CellDim, dims.KDim, low=250.0, high=300.0)
    input_state = tmx_states.TmxInputState(
        temperature=temperature(),
        virtual_temperature=temperature(),
        pressure=random_field(dims.CellDim, dims.KDim, low=1.0e4, high=1.0e5),
        u=wind(),
        v=wind(),
        w=random_field(dims.CellDim, dims.KHalfDim, low=-1.0, high=1.0),
        qv=tracer(),
        qc=tracer(),
        qi=tracer(),
        qr=tracer(),
        qs=tracer(),
        qg=tracer(),
        rho=random_field(dims.CellDim, dims.KDim, low=0.5, high=1.3),
        air_mass=random_field(dims.CellDim, dims.KDim, low=100.0, high=1000.0),
        cv_air=random_field(dims.CellDim, dims.KDim, low=7.0e4, high=8.0e5),
    )
    surface_flux_state = tmx_states.TmxSurfaceFluxState(
        evapotranspiration=random_field(dims.CellDim, low=-1.0e-4, high=1.0e-4),
        sensible_heat_flux=random_field(dims.CellDim, low=-100.0, high=100.0),
        u_stress=random_field(dims.CellDim, low=-0.1, high=0.1),
        v_stress=random_field(dims.CellDim, low=-0.1, high=0.1),
        q_snocpymlt=random_field(dims.CellDim, low=0.0, high=1.0),
    )
    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(grid, allocator=allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(grid, allocator=allocator)

    component.run(
        input_state=input_state,
        surface_flux_state=surface_flux_state,
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        new_state=new_state,
        dtime=300.0,
    )

    for state in (diagnostic_state, tendency_state, new_state):
        for name, field in vars(state).items():
            assert np.all(np.isfinite(field.asnumpy())), name
    for name, field in vars(tendency_state).items():
        assert np.any(field.asnumpy() != 0.0), name

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

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config, tmx, tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc

from ..fixtures import *  # noqa: F403
from .utils import random_static_states


def test_tmx_run_smoke(backend_like: model_backends.BackendLike) -> None:
    allocator = model_backends.get_allocator(backend_like)
    grid = simple.simple_grid(allocator=allocator, num_levels=10)
    random_field = functools.partial(data_alloc.random_field, grid, allocator=allocator)
    static_states = random_static_states(grid, allocator)
    component = tmx.Tmx(
        grid=grid,
        config=tmx_config.TmxConfig(),
        metric_state=static_states.metric_state,
        interpolation_state=static_states.interpolation_state,
        edge_params=static_states.edge_params,
        cell_params=static_states.cell_params,
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

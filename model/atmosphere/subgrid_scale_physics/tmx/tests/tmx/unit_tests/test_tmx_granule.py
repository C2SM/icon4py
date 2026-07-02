# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test of the Tmx granule on a synthetic grid.

Constructs the granule with plausible (random but physically sane) metric,
interpolation and geometry fields on the simple grid and runs the Stage A
diagnostics, the scalar diffusion stages B (hydrometeors) and C (temperature),
the momentum diffusion stages D (horizontal wind) and E (vertical wind), the
energy update F and the diagnostics update G once, followed by a full
``run()`` (Stages A to G). This only checks that the orchestration is wired
correctly (programs execute, outputs are finite and have the right shapes);
correctness against ICON is covered by the stencil tests and by the
integration datatests (``integration_tests/``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@pytest.fixture
def grid(backend_like: model_backends.BackendLike) -> base_grid.Grid:
    return simple.simple_grid(allocator=model_backends.get_allocator(backend_like), num_levels=10)


def _metric_state(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tmx_states.TmxMetricState:
    def positive(*dimensions, extend=None):
        return data_alloc.random_field(
            grid, *dimensions, low=0.1, high=1.0, extend=extend, allocator=allocator
        )

    def weight(*dimensions, extend=None):
        return data_alloc.random_field(
            grid, *dimensions, low=0.3, high=0.7, extend=extend, allocator=allocator
        )

    return tmx_states.TmxMetricState(
        ddqz_z_full=positive(dims.CellDim, dims.KDim),
        inv_ddqz_z_full=positive(dims.CellDim, dims.KDim),
        ddqz_z_half=positive(dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
        inv_ddqz_z_half=positive(dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
        inv_ddqz_z_full_e=positive(dims.EdgeDim, dims.KDim),
        inv_ddqz_z_half_e=positive(dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}),
        inv_ddqz_z_half_v=positive(dims.VertexDim, dims.KDim, extend={dims.KDim: 1}),
        wgtfac_c=weight(dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
        wgtfac_e=weight(dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}),
        wgtfacq_c=weight(dims.CellDim, dims.KDim),
        wgtfacq1_c=weight(dims.CellDim, dims.KDim),
        wgtfacq_e=weight(dims.EdgeDim, dims.KDim),
        wgtfacq1_e=weight(dims.EdgeDim, dims.KDim),
        geopot_agl_ifc=data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=100.0,
            high=1.0e5,
            extend={dims.KDim: 1},
            allocator=allocator,
        ),
        z_mc=positive(dims.CellDim, dims.KDim),
        z_ifc=positive(dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
        edge_cell_length=data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, low=1.0e3, high=1.0e4, allocator=allocator
        ),
    )


def _interpolation_state(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tmx_states.TmxInterpolationState:
    def coeff(*dimensions):
        return data_alloc.random_field(grid, *dimensions, low=0.1, high=0.5, allocator=allocator)

    return tmx_states.TmxInterpolationState(
        c_lin_e=coeff(dims.EdgeDim, dims.E2CDim),
        e_bln_c_s=coeff(dims.CellDim, dims.C2EDim),
        geofac_div=coeff(dims.CellDim, dims.C2EDim),
        cells_aw_verts=coeff(dims.VertexDim, dims.V2CDim),
        rbf_coeff_v1=coeff(dims.VertexDim, dims.V2EDim),
        rbf_coeff_v2=coeff(dims.VertexDim, dims.V2EDim),
        rbf_coeff_e=coeff(dims.EdgeDim, dims.E2C2EDim),
        rbf_coeff_c1=coeff(dims.CellDim, dims.C2E2C2EDim),
        rbf_coeff_c2=coeff(dims.CellDim, dims.C2E2C2EDim),
    )


def _geometry(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tuple[grid_states.EdgeParams, grid_states.CellParams]:
    def edge_field(*dimensions):
        return data_alloc.random_field(grid, *dimensions, low=-1.0, high=1.0, allocator=allocator)

    def positive_edge_field():
        return data_alloc.random_field(
            grid, dims.EdgeDim, low=1.0e-5, high=1.0e-3, allocator=allocator
        )

    edge_params = grid_states.EdgeParams(
        tangent_orientation=edge_field(dims.EdgeDim),
        inverse_primal_edge_lengths=positive_edge_field(),
        inverse_dual_edge_lengths=positive_edge_field(),
        inverse_vertex_vertex_lengths=positive_edge_field(),
        primal_normal_vert_x=edge_field(dims.EdgeDim, dims.E2C2VDim),
        primal_normal_vert_y=edge_field(dims.EdgeDim, dims.E2C2VDim),
        dual_normal_vert_x=edge_field(dims.EdgeDim, dims.E2C2VDim),
        dual_normal_vert_y=edge_field(dims.EdgeDim, dims.E2C2VDim),
        primal_normal_cell_x=edge_field(dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_y=edge_field(dims.EdgeDim, dims.E2CDim),
        dual_normal_cell_x=edge_field(dims.EdgeDim, dims.E2CDim),
        dual_normal_cell_y=edge_field(dims.EdgeDim, dims.E2CDim),
    )
    cell_params = grid_states.CellParams(
        area=data_alloc.random_field(grid, dims.CellDim, low=1.0e6, high=1.0e8, allocator=allocator)
    )
    return edge_params, cell_params


def _input_state(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tmx_states.TmxInputState:
    def wind():
        return data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=-10.0, high=10.0, allocator=allocator
        )

    def tracer():
        return data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, allocator=allocator
        )

    return tmx_states.TmxInputState(
        temperature=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=250.0, high=300.0, allocator=allocator
        ),
        virtual_temperature=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=250.0, high=300.0, allocator=allocator
        ),
        pressure=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e4, high=1.0e5, allocator=allocator
        ),
        pressure_ifc=data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=1.0e4,
            high=1.0e5,
            extend={dims.KDim: 1},
            allocator=allocator,
        ),
        u=wind(),
        v=wind(),
        w=data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=-1.0,
            high=1.0,
            extend={dims.KDim: 1},
            allocator=allocator,
        ),
        qv=tracer(),
        qc=tracer(),
        qi=tracer(),
        qr=tracer(),
        qs=tracer(),
        qg=tracer(),
        rho=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.5, high=1.3, allocator=allocator
        ),
        air_mass=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=100.0, high=1000.0, allocator=allocator
        ),
        cv_air=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=700.0, high=800.0, allocator=allocator
        ),
    )


def _surface_flux_state(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tmx_states.TmxSurfaceFluxState:
    def flux(low: float, high: float):
        return data_alloc.random_field(grid, dims.CellDim, low=low, high=high, allocator=allocator)

    return tmx_states.TmxSurfaceFluxState(
        evapotranspiration=flux(-1.0e-4, 1.0e-4),
        sensible_heat_flux=flux(-100.0, 100.0),
        u_stress=flux(-0.1, 0.1),
        v_stress=flux(-0.1, 0.1),
        q_snocpymlt=flux(0.0, 1.0),
    )


#: diagnostic-state fields written by run_diagnostics and their expected
#: (horizontal dimension, number of vertical levels relative to nlev) shapes
STAGE_A_OUTPUTS = (
    ("cptgz", dims.CellDim, 0),
    ("theta_v", dims.CellDim, 0),
    ("rho_ic", dims.CellDim, 1),
    ("bruvais", dims.CellDim, 1),
    ("vn", dims.EdgeDim, 0),
    ("w_vert", dims.VertexDim, 1),
    ("w_ie", dims.EdgeDim, 1),
    ("u_vert", dims.VertexDim, 0),
    ("v_vert", dims.VertexDim, 0),
    ("vn_ie", dims.EdgeDim, 1),
    ("vt_ie", dims.EdgeDim, 1),
    ("shear", dims.EdgeDim, 0),
    ("div_of_stress", dims.EdgeDim, 0),
    ("div_c", dims.CellDim, 0),
    ("mech_prod", dims.CellDim, 1),
    ("km_ic", dims.CellDim, 1),
    ("kh_ic", dims.CellDim, 1),
    ("km_c", dims.CellDim, 0),
    ("km_iv", dims.VertexDim, 1),
    ("km_ie", dims.EdgeDim, 1),
)


def test_tmx_granule_construction_and_diagnostics_smoke(
    grid: base_grid.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    config = tmx.TmxConfig()
    params = tmx.TmxParams(config)
    edge_params, cell_params = _geometry(grid, allocator)

    granule = tmx.Tmx(
        grid=grid,
        config=config,
        params=params,
        vertical_grid=None,
        metric_state=_metric_state(grid, allocator),
        interpolation_state=_interpolation_state(grid, allocator),
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend_like,
    )

    # init fields (computed in __init__)
    mix_len_sq = granule.mix_len_sq.asnumpy()
    assert mix_len_sq.shape == (grid.num_cells, grid.num_levels + 1)
    assert np.all(np.isfinite(mix_len_sq)) and np.all(mix_len_sq >= 0.0)
    louis_factor = granule.louis_factor.asnumpy()
    assert louis_factor.shape == (grid.num_cells,)
    assert np.all(louis_factor > 0.0)
    ghf = granule.ghf.asnumpy()
    assert ghf.shape == (grid.num_cells, grid.num_levels)
    assert np.all(np.isfinite(ghf))

    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(grid, allocator=allocator)
    input_state = _input_state(grid, allocator)
    granule.run_diagnostics(input_state, diagnostic_state)

    horizontal_size = {
        dims.CellDim: grid.num_cells,
        dims.EdgeDim: grid.num_edges,
        dims.VertexDim: grid.num_vertices,
    }
    for name, horizontal_dim, vertical_extend in STAGE_A_OUTPUTS:
        field = getattr(diagnostic_state, name).asnumpy()
        assert field.shape == (
            horizontal_size[horizontal_dim],
            grid.num_levels + vertical_extend,
        ), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"

    # the viscosities have a positivity guarantee (floor km_min after interpolation)
    assert np.all(diagnostic_state.km_c.asnumpy() >= config.km_min)
    assert np.all(diagnostic_state.km_iv.asnumpy() >= config.km_min)
    assert np.all(diagnostic_state.km_ie.asnumpy() >= config.km_min)

    # scalar diffusion stages B (hydrometeors) and C (temperature)
    surface_flux_state = _surface_flux_state(grid, allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(grid, allocator=allocator)
    dtime = 300.0

    granule.run_hydrometeor_diffusion(
        input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime
    )
    granule.run_temperature_diffusion(
        input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime
    )

    for name in ("ddt_qv", "ddt_qc", "ddt_qi", "ddt_temperature"):
        field = getattr(tendency_state, name).asnumpy()
        assert field.shape == (grid.num_cells, grid.num_levels), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    for name in ("qv", "qc", "qi", "temperature"):
        field = getattr(new_state, name).asnumpy()
        assert field.shape == (grid.num_cells, grid.num_levels), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    assert np.all(np.isfinite(granule.energy.asnumpy()))
    assert np.all(np.isfinite(granule.tend_energy.asnumpy()))

    # momentum diffusion stages D (horizontal wind) and E (vertical wind)
    granule.run_horizontal_wind_diffusion(
        input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime
    )
    granule.run_vertical_wind_diffusion(
        input_state, diagnostic_state, tendency_state, new_state, dtime
    )

    tot_tend = granule.tot_tend.asnumpy()
    assert tot_tend.shape == (grid.num_edges, grid.num_levels)
    assert np.all(np.isfinite(tot_tend))
    for name in ("ddt_u", "ddt_v"):
        field = getattr(tendency_state, name).asnumpy()
        assert field.shape == (grid.num_cells, grid.num_levels), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    for name in ("u", "v"):
        field = getattr(new_state, name).asnumpy()
        assert field.shape == (grid.num_cells, grid.num_levels), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    ddt_w = tendency_state.ddt_w.asnumpy()
    assert ddt_w.shape == (grid.num_cells, grid.num_levels + 1)
    assert np.all(np.isfinite(ddt_w))
    new_w = new_state.w.asnumpy()
    assert new_w.shape == (grid.num_cells, grid.num_levels + 1)
    assert np.all(np.isfinite(new_w))
    # w = 0 boundary conditions: the top and bottom half-level rows of the new
    # w and its tendency are never written
    np.testing.assert_array_equal(ddt_w[:, 0], 0.0)
    np.testing.assert_array_equal(ddt_w[:, -1], 0.0)
    np.testing.assert_array_equal(new_w[:, 0], 0.0)
    np.testing.assert_array_equal(new_w[:, -1], 0.0)

    # energy update stage F and diagnostics update stage G
    granule.run_energy_update(
        input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime
    )
    granule.run_update_diagnostics(input_state, diagnostic_state, new_state, dtime)

    for name in ("heating", "dissip_ke", "cptgz", "km", "kh"):
        field = getattr(diagnostic_state, name).asnumpy()
        assert field.shape == (grid.num_cells, grid.num_levels), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    for name in ("cptgz_vi", "dissip_ke_vi", "int_energy_vi", "int_energy_vi_tend"):
        field = getattr(diagnostic_state, name).asnumpy()
        assert field.shape == (grid.num_cells,), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    # the km/kh diagnostics are the interior half-level coefficients shifted
    # by one row; the bottom row is zero (the surface exchange coefficients
    # are out of scope and use_km_const is False by default)
    km = diagnostic_state.km.asnumpy()
    kh = diagnostic_state.kh.asnumpy()
    np.testing.assert_array_equal(km[:, :-1], diagnostic_state.km_ic.asnumpy()[:, 1:-1])
    np.testing.assert_array_equal(kh[:, :-1], diagnostic_state.kh_ic.asnumpy()[:, 1:-1])
    np.testing.assert_array_equal(km[:, -1], 0.0)
    np.testing.assert_array_equal(kh[:, -1], 0.0)


def test_tmx_granule_full_run_smoke(
    grid: base_grid.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    """Run the full A-to-G sequence through ``Tmx.run`` once."""
    allocator = model_backends.get_allocator(backend_like)
    config = tmx.TmxConfig()
    edge_params, cell_params = _geometry(grid, allocator)

    granule = tmx.Tmx(
        grid=grid,
        config=config,
        params=tmx.TmxParams(config),
        vertical_grid=None,
        metric_state=_metric_state(grid, allocator),
        interpolation_state=_interpolation_state(grid, allocator),
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend_like,
    )

    input_state = _input_state(grid, allocator)
    surface_flux_state = _surface_flux_state(grid, allocator)
    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(grid, allocator=allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(grid, allocator=allocator)

    granule.run(
        input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime=300.0
    )

    for state, names in (
        (tendency_state, ("ddt_temperature", "ddt_qv", "ddt_qc", "ddt_qi", "ddt_u", "ddt_v")),
        (new_state, ("temperature", "qv", "qc", "qi", "u", "v")),
    ):
        for name in names:
            field = getattr(state, name).asnumpy()
            assert field.shape == (
                grid.num_cells,
                grid.num_levels,
            ), f"unexpected shape for '{name}'"
            assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    for state, name in ((tendency_state, "ddt_w"), (new_state, "w")):
        field = getattr(state, name).asnumpy()
        assert field.shape == (grid.num_cells, grid.num_levels + 1)
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"
    for name in ("heating", "dissip_ke", "km", "kh"):
        assert np.all(np.isfinite(getattr(diagnostic_state, name).asnumpy())), (
            f"non-finite values in '{name}'"
        )
    for name in ("cptgz_vi", "dissip_ke_vi", "int_energy_vi", "int_energy_vi_tend"):
        field = getattr(diagnostic_state, name).asnumpy()
        assert field.shape == (grid.num_cells,), f"unexpected shape for '{name}'"
        assert np.all(np.isfinite(field)), f"non-finite values in '{name}'"

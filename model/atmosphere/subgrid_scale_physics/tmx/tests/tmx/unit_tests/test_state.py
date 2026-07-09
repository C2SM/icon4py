# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.state import TmxState
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.states import prognostic_state as prognostics, tracer_state
from icon4py.model.common.utils import data_allocation as data_alloc


# ---------------------------------------------------------------------------
# Helper factories (adapted from the muphys test_state pattern)
# ---------------------------------------------------------------------------


def _uniform_prognostic(
    grid,
    *,
    rho: float = 1.2,
    exner: float = 0.95,
    theta_v: float = 300.0,
) -> prognostics.PrognosticState:
    """PrognosticState filled with uniform constant values on the simple grid."""
    return prognostics.PrognosticState(
        rho=data_alloc.constant_field(grid, rho, dims.CellDim, dims.KDim),
        w=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
        vn=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
        exner=data_alloc.constant_field(grid, exner, dims.CellDim, dims.KDim),
        theta_v=data_alloc.constant_field(grid, theta_v, dims.CellDim, dims.KDim),
    )


def _tracer_state(
    grid,
    *,
    qv: float = 0.0,
    qc: float = 0.0,
    qi: float = 0.0,
    qr: float = 0.0,
    qs: float = 0.0,
    qg: float = 0.0,
) -> tracer_state.TracerState:
    """TracerState with all six species active (defaults to zero except where specified)."""
    return tracer_state.TracerState(
        qv=data_alloc.constant_field(grid, qv, dims.CellDim, dims.KDim),
        qc=data_alloc.constant_field(grid, qc, dims.CellDim, dims.KDim),
        qi=data_alloc.constant_field(grid, qi, dims.CellDim, dims.KDim),
        qr=data_alloc.constant_field(grid, qr, dims.CellDim, dims.KDim),
        qs=data_alloc.constant_field(grid, qs, dims.CellDim, dims.KDim),
        qg=data_alloc.constant_field(grid, qg, dims.CellDim, dims.KDim),
    )


def _tmx_state(grid) -> TmxState:
    """Construct a TmxState on the simple grid with neutral/zero interpolation coefficients."""
    return TmxState(
        grid=grid,
        ddqz_z_full=data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim),
        rbf_coeff_c1=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
        rbf_coeff_c2=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
        c_lin_e=data_alloc.constant_field(grid, 0.5, dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_x=data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_y=data_alloc.zero_field(grid, dims.EdgeDim, dims.E2CDim),
        backend=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_as_component_input_matches_contract():
    """as_component_input must return exactly the 21 keys of INPUTS_PROPERTIES."""
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    inp = state.as_component_input()
    assert set(inp) == set(tmx_data.INPUTS_PROPERTIES)


def test_gather_computes_air_mass_and_zero_surface_fluxes():
    """air_mass == rho * ddqz_z_full; surface-flux buffers stay zero before TMX runs."""
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    inp = state.as_component_input()
    np.testing.assert_allclose(
        inp["air_mass"].asnumpy(), prognostic.rho.asnumpy() * 100.0, rtol=1e-14
    )
    for key in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress", "q_snocpymlt"):
        assert (inp[key].asnumpy() == 0.0).all()


# ---------------------------------------------------------------------------
# Task-5 helpers and tests: scatter_to_prognostic
# ---------------------------------------------------------------------------


def _tmx_outputs(grid, *, ddt_u=0.0, ddt_v=0.0, ddt_w=0.0, ddt_qv=0.0):
    def ck(value, **kw):
        return data_alloc.constant_field(grid, value, dims.CellDim, dims.KDim, **kw)

    # ddt_w spans KDim+1 half-levels; constant_field does not support 'extend',
    # so we use zero_field (which does) and fill the backing array.
    _ddt_w = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
    if ddt_w != 0.0:
        _ddt_w.ndarray[:] = ddt_w

    out = {
        "ddt_temperature": ck(0.0),
        "ddt_qv": ck(ddt_qv), "ddt_qc": ck(0.0), "ddt_qi": ck(0.0),
        "ddt_u": ck(ddt_u), "ddt_v": ck(ddt_v),
        "ddt_w": _ddt_w,
        "km": ck(0.0), "kh": ck(0.0), "heating": ck(0.0), "dissip_ke": ck(0.0),
    }
    for key in ("cptgz_vi", "dissip_ke_vi", "int_energy_vi", "int_energy_vi_tend"):
        out[key] = data_alloc.constant_field(grid, 0.0, dims.CellDim)
    return out


def test_scatter_applies_qv_and_w_tendencies():
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    dt = 300.0
    state.scatter_to_prognostic(
        prognostic, _tmx_outputs(grid, ddt_qv=1e-7, ddt_w=1e-4), datetime.timedelta(seconds=dt)
    )
    np.testing.assert_allclose(tracers.qv.asnumpy(), 1e-3 + 1e-7 * dt, rtol=1e-12)
    np.testing.assert_allclose(prognostic.w.asnumpy(), 1e-4 * dt, rtol=1e-12)


def test_scatter_projects_wind_tendency_to_vn():
    # uniform ddt_u = 1e-4, ddt_v = 0; primal_normal_cell_x = 1, c_lin_e = 0.5 (two neighbors)
    # => ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on interior edges
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    dt = 300.0
    state.scatter_to_prognostic(
        prognostic, _tmx_outputs(grid, ddt_u=1e-4), datetime.timedelta(seconds=dt)
    )
    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * dt, rtol=1e-12)

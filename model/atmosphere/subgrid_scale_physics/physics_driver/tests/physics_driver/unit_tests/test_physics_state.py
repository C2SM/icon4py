# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the PhysicsState layer (EntryState facade, accumulators, apply-once)."""

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import geometry_attributes, simple
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


# ---------------------------------------------------------------------------
# Helper factories (same simple_grid + stub-source pattern as the tmx state tests)
# ---------------------------------------------------------------------------


class _StubFieldSource:
    """Minimal FieldSource stand-in: serves pre-built fields by attribute name."""

    def __init__(self, fields):
        self._fields = fields

    def get(self, name, *args, **kwargs):
        return self._fields[name]


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


def _tracer_state(grid, *, qv: float = 0.0) -> tracer_states.TracerState:
    """TracerState with all six species active."""
    ck = lambda value: data_alloc.constant_field(grid, value, dims.CellDim, dims.KDim)  # noqa: E731
    return tracer_states.TracerState(
        qv=ck(qv), qc=ck(0.0), qi=ck(0.0), qr=ck(0.0), qs=ck(0.0), qg=ck(0.0)
    )


def _entry_state(grid) -> physics_state.EntryState:
    metrics = _StubFieldSource(
        {
            metrics_attributes.DDQZ_Z_FULL: data_alloc.constant_field(
                grid, 100.0, dims.CellDim, dims.KDim
            ),
        }
    )
    interpolation = _StubFieldSource(
        {
            interpolation_attributes.RBF_VEC_COEFF_C1: data_alloc.zero_field(
                grid, dims.CellDim, dims.C2E2C2EDim
            ),
            interpolation_attributes.RBF_VEC_COEFF_C2: data_alloc.zero_field(
                grid, dims.CellDim, dims.C2E2C2EDim
            ),
        }
    )
    return physics_state.EntryState(
        grid=grid,
        interpolation=interpolation,
        metrics=metrics,
        backend=None,
    )


# ---------------------------------------------------------------------------
# EntryState
# ---------------------------------------------------------------------------


def test_diagnose_from_fills_working_fields_and_leaves_inputs_untouched():
    """The dyn2phy diagnosis fills plausible fields and is strictly read-only.

    Read-only-ness is the load-bearing invariant of parallel coupling: the
    prognostic state and tracers must stay bitwise identical until the driver's
    single apply step.
    """
    grid = simple.simple_grid()
    ws = _entry_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    exner_before = prognostic.exner.asnumpy().copy()
    vn_before = prognostic.vn.asnumpy().copy()

    ws.diagnose_from(prognostic, tracers)

    # wiring smoke test: physically plausible diagnostics
    assert 200.0 < ws.ta.asnumpy().mean() < 320.0
    assert (ws.pressure.asnumpy() > 0).all()
    # pressure grows downward: surface interface > top full level
    assert (ws.pressure_ifc.asnumpy()[:, -1] > ws.pressure.asnumpy()[:, 0]).all()

    # the invariant: inputs untouched
    np.testing.assert_array_equal(prognostic.exner.asnumpy(), exner_before)
    np.testing.assert_array_equal(prognostic.vn.asnumpy(), vn_before)
    np.testing.assert_allclose(tracers.qv.asnumpy(), 1e-3, rtol=0)

    # the facade binds pointers, not copies: same objects, physics names
    assert ws.exner is prognostic.exner
    assert ws.rho is prognostic.rho
    assert ws.w is prognostic.w
    assert ws.vn is prognostic.vn
    assert ws.tracers is tracers


# ---------------------------------------------------------------------------
# TendencyAccumulators
# ---------------------------------------------------------------------------


def test_accumulate_sums_tendencies_and_skips_diagnostics():
    grid = simple.simple_grid()
    acc = physics_state.TendencyAccumulators()
    props = {"tend_qv": {"kind": "tendency"}, "km": {"kind": "diagnostic"}}
    out = {
        "tend_qv": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim),
        "km": data_alloc.constant_field(grid, 5.0, dims.CellDim, dims.KDim),
    }

    acc.zero()
    acc.accumulate(out, props)
    acc.accumulate(out, props)  # a second process contributing the same tendency

    np.testing.assert_allclose(acc.acc["tend_qv"].asnumpy(), 2e-7, rtol=1e-12)
    assert "km" not in acc.acc


def test_zero_resets_between_steps():
    grid = simple.simple_grid()
    acc = physics_state.TendencyAccumulators()
    props = {"tend_qv": {"kind": "tendency"}}
    out = {"tend_qv": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim)}

    acc.zero()
    acc.accumulate(out, props)
    acc.zero()
    acc.accumulate(out, props)

    np.testing.assert_allclose(acc.acc["tend_qv"].asnumpy(), 1e-7, rtol=1e-12)


# ---------------------------------------------------------------------------
# ApplyToPrognostic
# ---------------------------------------------------------------------------


def _apply_to_prognostic(grid) -> physics_state.ApplyToPrognostic:
    # neutral geometry: primal_normal_x = 1, primal_normal_y = 0, c_lin_e = 0.5
    # => two-neighbor projection of a uniform u-tendency is the identity
    geometry = _StubFieldSource(
        {
            geometry_attributes.EDGE_NORMAL_CELL_U: data_alloc.constant_field(
                grid, 1.0, dims.EdgeDim, dims.E2CDim
            ),
            geometry_attributes.EDGE_NORMAL_CELL_V: data_alloc.zero_field(
                grid, dims.EdgeDim, dims.E2CDim
            ),
        }
    )
    interpolation = _StubFieldSource(
        {
            interpolation_attributes.C_LIN_E: data_alloc.constant_field(
                grid, 0.5, dims.EdgeDim, dims.E2CDim
            ),
        }
    )
    return physics_state.ApplyToPrognostic(
        grid=grid, geometry=geometry, interpolation=interpolation, backend=None
    )


def _accumulated(grid, **tendencies) -> physics_state.TendencyAccumulators:
    """Accumulators pre-filled with the given constant tendencies (single process)."""
    acc = physics_state.TendencyAccumulators()
    props = {name: {"kind": "tendency"} for name in tendencies}
    acc.zero()
    acc.accumulate(tendencies, props)
    return acc


def test_apply_updates_tracers_w_and_thermodynamics_once():
    grid = simple.simple_grid()
    ws = _entry_state(grid)
    apply_once = _apply_to_prognostic(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    ws.diagnose_from(prognostic, tracers)
    exner_before = prognostic.exner.asnumpy().copy()
    theta_v_before = prognostic.theta_v.asnumpy().copy()

    # ddt_w spans KDim+1 half-levels; constant_field does not support 'extend',
    # so we use zero_field (which does) and fill the backing array.
    tend_w = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
    tend_w.ndarray[...] = 1e-4
    dt = 300.0
    acc = _accumulated(
        grid,
        tend_qv=data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim),
        tend_temperature=data_alloc.constant_field(grid, 1e-3, dims.CellDim, dims.KDim),
        tend_w=tend_w,
    )

    apply_once(ws, acc, dt_seconds=dt)

    np.testing.assert_allclose(tracers.qv.asnumpy(), 1e-3 + 1e-7 * dt, rtol=1e-12)
    np.testing.assert_allclose(prognostic.w.asnumpy(), 1e-4 * dt, rtol=1e-12)
    # EOS wiring smoke test: the exact-EOS update must have rewritten exner and
    # theta_v (their new values are EOS-consistent with rho and the updated Tv;
    # no direction assertion — the uniform test state is not EOS-consistent).
    assert not np.array_equal(prognostic.exner.asnumpy(), exner_before)
    assert not np.array_equal(prognostic.theta_v.asnumpy(), theta_v_before)


def test_apply_projects_accumulated_wind_tendency_to_vn():
    # uniform ddt_u = 1e-4, ddt_v = 0; primal_normal_cell_x = 1, c_lin_e = 0.5 (two neighbors)
    # => ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on all edges of the periodic simple grid
    grid = simple.simple_grid()
    ws = _entry_state(grid)
    apply_once = _apply_to_prognostic(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    ws.diagnose_from(prognostic, tracers)
    dt = 300.0
    acc = _accumulated(
        grid,
        tend_u=data_alloc.constant_field(grid, 1e-4, dims.CellDim, dims.KDim),
        tend_v=data_alloc.zero_field(grid, dims.CellDim, dims.KDim),
    )

    apply_once(ws, acc, dt_seconds=dt)

    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * dt, rtol=1e-12)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the PhysicsState layer (PhysicsState facade, process outputs, tendency sums, single update)."""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry_attributes, simple
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import (
    diagnostic_state,
    model,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc


def _meta(*field_dims: gtx.Dimension, kind: model.FieldKind | None = None) -> model.FieldMetaData:
    """Output metadata for the routing tests: only ``kind`` and ``dims`` are read.

    No ``kind`` is how a diagnostic output declares itself — the routing keys off
    ``TENDENCY`` alone.
    """
    return model.FieldMetaData(
        standard_name="test_field", units="1", kind=kind, dims=field_dims or None
    )


_TENDENCY = model.FieldKind.TENDENCY
_DT_SECONDS = 300.0


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
        w=data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim),
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


def _diagnostic_state(grid) -> diagnostic_state.DiagnosticState:
    """The caller-owned DiagnosticState that compute_diagnostics links and fills."""
    return diagnostic_state.initialize_diagnostic_state(grid, None)


class _RecordingExchange:
    """Records the halo exchanges as (dim, field identities), so their order and targets are assertable."""

    def __init__(self) -> None:
        self.events: list[tuple[gtx.Dimension, list[int]]] = []

    def exchange(self, dim: gtx.Dimension, *fields: gtx.Field) -> None:
        self.events.append((dim, [id(f) for f in fields]))


def _physics_state(grid, exchange=None) -> physics_state.PhysicsState:
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
            interpolation_attributes.C_LIN_E: data_alloc.constant_field(
                grid, 0.5, dims.EdgeDim, dims.E2CDim
            ),
        }
    )
    return physics_state.PhysicsState(
        grid=grid,
        geometry=geometry,
        interpolation=interpolation,
        metrics=metrics,
        exchange=exchange if exchange is not None else decomposition.SingleNodeExchange(),
        backend=None,
    )


# ---------------------------------------------------------------------------
# PhysicsState
# ---------------------------------------------------------------------------


def test_diagnose_fills_working_fields_and_leaves_inputs_untouched():
    """The dyn2phy diagnosis fills plausible fields and is strictly read-only.

    The prognostic state and tracers must stay bitwise identical: only
    ``advance`` and ``update_prognostics`` change them.
    """
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    exner_before = prognostic.exner.asnumpy().copy()
    vn_before = prognostic.vn.asnumpy().copy()

    ws.compute_diagnostics(prognostic, tracers, _diagnostic_state(grid))

    # wiring smoke test: physically plausible diagnostics
    assert 200.0 < ws.diagnostics.temperature.asnumpy().mean() < 320.0
    assert (ws.diagnostics.pressure.asnumpy() > 0).all()
    # pressure grows downward: surface interface > top full level
    assert (
        ws.diagnostics.pressure_ifc.asnumpy()[:, -1] > ws.diagnostics.pressure.asnumpy()[:, 0]
    ).all()

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
# Summing the tendencies
# ---------------------------------------------------------------------------


def test_accumulate_sums_tendencies_and_skips_diagnostics():
    grid = simple.simple_grid()
    acc = _tendencies()
    props = {"tend_qv": _meta(kind=_TENDENCY), "km": _meta()}
    out = {
        "tend_qv": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim),
        "km": data_alloc.constant_field(grid, 5.0, dims.CellDim, dims.KDim),
    }

    acc.zero()
    acc.accumulate(out, props)
    acc.accumulate(out, props)  # a second process contributing the same tendency

    np.testing.assert_allclose(acc.sums["tend_qv"].asnumpy(), 2e-7, rtol=1e-12)
    assert "km" not in acc.sums


def test_zero_resets_between_steps():
    grid = simple.simple_grid()
    acc = _tendencies()
    props = {"tend_qv": _meta(kind=_TENDENCY)}
    out = {"tend_qv": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim)}

    acc.zero()
    acc.accumulate(out, props)
    acc.zero()
    acc.accumulate(out, props)

    np.testing.assert_allclose(acc.sums["tend_qv"].asnumpy(), 1e-7, rtol=1e-12)


# ---------------------------------------------------------------------------
# Updating the state with the tendencies
# ---------------------------------------------------------------------------


def _tendencies() -> physics_state.Tendencies:
    return physics_state.Tendencies(backend=None)


def _accumulated(grid, state, **tendencies) -> physics_state.Tendencies:
    """Tendencies of one process with the given constant tendencies, summed and advanced into ``state``."""
    acc = _tendencies()
    props = {name: _meta(kind=_TENDENCY) for name in tendencies}
    acc.zero()
    acc.accumulate(tendencies, props)
    state.advance(tendencies, props, _DT_SECONDS)
    return acc


def test_advance_and_update_apply_tracers_w_and_thermodynamics_once():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    ws.compute_diagnostics(prognostic, tracers, _diagnostic_state(grid))
    exner_before = prognostic.exner.asnumpy().copy()
    theta_v_before = prognostic.theta_v.asnumpy().copy()

    tend_w = data_alloc.constant_field(grid, 1e-4, dims.CellDim, dims.KHalfDim)
    dt = 300.0
    acc = _accumulated(
        grid,
        ws,
        tend_qv=data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim),
        tend_temperature=data_alloc.constant_field(grid, 1e-3, dims.CellDim, dims.KDim),
        tend_w=tend_w,
    )

    ws.update_prognostics(acc.sums, dt_seconds=dt)

    np.testing.assert_allclose(tracers.qv.asnumpy(), 1e-3 + 1e-7 * dt, rtol=1e-12)
    np.testing.assert_allclose(prognostic.w.asnumpy(), 1e-4 * dt, rtol=1e-12)
    # EOS wiring smoke test: the exact-EOS update must have rewritten exner and
    # theta_v (their new values are EOS-consistent with rho and the updated Tv;
    # no direction assertion — the uniform test state is not EOS-consistent).
    assert not np.array_equal(prognostic.exner.asnumpy(), exner_before)
    assert not np.array_equal(prognostic.theta_v.asnumpy(), theta_v_before)


def test_update_projects_accumulated_wind_tendency_to_vn():
    # uniform ddt_u = 1e-4, ddt_v = 0; primal_normal_cell_x = 1, c_lin_e = 0.5 (two neighbors)
    # => ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on all edges of the periodic simple grid
    grid = simple.simple_grid()
    exchange = _RecordingExchange()
    ws = _physics_state(grid, exchange)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    ws.compute_diagnostics(prognostic, tracers, _diagnostic_state(grid))
    dt = 300.0
    acc = _accumulated(
        grid,
        ws,
        tend_u=data_alloc.constant_field(grid, 1e-4, dims.CellDim, dims.KDim),
        tend_v=data_alloc.zero_field(grid, dims.CellDim, dims.KDim),
    )

    ws.update_prognostics(acc.sums, dt_seconds=dt)

    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * dt, rtol=1e-12)
    # the summed wind tendencies are synced before the projection, the written cells and vn after it
    assert [dim for dim, _ in exchange.events] == [dims.CellDim, dims.CellDim, dims.EdgeDim]
    assert exchange.events[0][1] == [id(acc.sums["tend_u"]), id(acc.sums["tend_v"])]
    assert exchange.events[-1][1] == [id(prognostic.vn)]


def test_update_prognostics_rejects_a_lone_wind_tendency():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), _diagnostic_state(grid))
    acc = _accumulated(
        grid, ws, tend_u=data_alloc.constant_field(grid, 1e-4, dims.CellDim, dims.KDim)
    )

    with pytest.raises(ValueError, match="applied as a pair; got only tend_u"):
        ws.update_prognostics(acc.sums, dt_seconds=_DT_SECONDS)


def test_sums_are_allocated_on_the_first_contribution():
    grid = simple.simple_grid()
    acc = _tendencies()
    assert acc.sums == {}

    acc.zero()
    acc.accumulate(
        {
            "tend_w": data_alloc.constant_field(grid, 1e-4, dims.CellDim, dims.KHalfDim),
            "tend_temperature": data_alloc.constant_field(grid, 1e-3, dims.CellDim, dims.KDim),
        },
        {"tend_w": _meta(kind=_TENDENCY), "tend_temperature": _meta(kind=_TENDENCY)},
    )

    # exactly the tendencies produced, shaped like them: w lives on half levels
    assert set(acc.sums) == {"tend_w", "tend_temperature"}
    assert acc.sums["tend_w"].domain.dims == (dims.CellDim, dims.KHalfDim)
    assert acc.sums["tend_temperature"].domain.dims == (dims.CellDim, dims.KDim)
    np.testing.assert_allclose(acc.sums["tend_w"].asnumpy(), 1e-4, rtol=1e-12)


def test_advance_rejects_a_tendency_of_no_advanced_field():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), _diagnostic_state(grid))
    out = {"tend_dummy": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim)}

    with pytest.raises(ValueError, match="'tend_dummy' is not the tendency"):
        ws.advance(out, {"tend_dummy": _meta(kind=_TENDENCY)}, _DT_SECONDS)


def test_physics_state_links_the_callers_diagnostic_state():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    diagnostics = _diagnostic_state(grid)
    temperature = diagnostics.temperature

    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), diagnostics)

    # linked, not copied: the caller's fields are the ones diagnosed and advanced
    assert ws.diagnostics is diagnostics
    assert 200.0 < temperature.asnumpy().mean() < 320.0
    # the flat shorthand is gone -- pointers and diagnostics are structurally distinct
    assert not hasattr(ws, "ta")
    assert not hasattr(ws, "pressure")


# ---------------------------------------------------------------------------
# Process outputs
# ---------------------------------------------------------------------------


def test_allocate_process_diagnostics_from_metadata():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    props = {
        "tend_temperature": _meta(dims.CellDim, dims.KDim, kind=_TENDENCY),
        "kh": _meta(dims.CellDim, dims.KDim),
        "cptgz_vi": _meta(dims.CellDim),
    }
    buffers = ws.allocate_process_diagnostics("tmx", props)
    # tendencies are never layer-allocated; diagnostics get their declared shape
    assert set(buffers) == {"kh", "cptgz_vi"}
    assert buffers["kh"].domain.dims == (dims.CellDim, dims.KDim)
    assert buffers["cptgz_vi"].domain.dims == (dims.CellDim,)
    assert ws.diagnostic_output("kh") is buffers["kh"]


def test_allocate_process_diagnostics_of_a_half_level_output_from_khalfdim():
    # Half levels are stated by the dimension alone, so a KHalfDim output must get
    # one level more than a full-level one.
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    buffers = ws.allocate_process_diagnostics("tmx", {"kh": _meta(dims.CellDim, dims.KHalfDim)})

    assert buffers["kh"].domain.dims == (dims.CellDim, dims.KHalfDim)
    assert buffers["kh"].ndarray.shape == (grid.num_cells, grid.num_levels + 1)


def test_allocate_process_diagnostics_rejects_a_duplicate_process_name():
    # A second allocation under the same name would silently orphan the buffers
    # already bound to the first process's component.
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    props = {"kh": _meta(dims.CellDim, dims.KDim)}
    ws.allocate_process_diagnostics("tmx", props)

    with pytest.raises(ValueError, match="already has output buffers"):
        ws.allocate_process_diagnostics("tmx", props)


def test_allocate_process_diagnostics_rejects_a_diagnostic_without_dims():
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    with pytest.raises(ValueError, match="declares no 'dims'"):
        ws.allocate_process_diagnostics("tmx", {"kh": _meta()})


def test_process_diagnostics_are_not_set_as_attributes():
    # The outputs stay behind 'process_diagnostics', so an output name can never shadow
    # a model-state pointer or the diagnostics.
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    ws.allocate_process_diagnostics("tmx", {"kh": _meta(dims.CellDim, dims.KDim)})

    assert not hasattr(ws, "kh")


# ---------------------------------------------------------------------------
# Reading another process's diagnostics
# ---------------------------------------------------------------------------


def test_diagnostic_output_finds_the_providing_process():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    surface = ws.allocate_process_diagnostics(
        "surface", {"sensible_heat_flux": _meta(dims.CellDim)}
    )
    ws.allocate_process_diagnostics("tmx", {"kh": _meta(dims.CellDim, dims.KDim)})

    # identity: the reader gets the very buffer the surface writes, by name alone
    assert ws.diagnostic_output("sensible_heat_flux") is surface["sensible_heat_flux"]
    with pytest.raises(KeyError, match="No process provides the diagnostic 'u_stress'"):
        ws.diagnostic_output("u_stress")


def test_allocate_process_diagnostics_rejects_a_second_provider_of_a_name():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.allocate_process_diagnostics("surface", {"surface_temperature": _meta(dims.CellDim)})

    with pytest.raises(ValueError, match="already provided by process 'surface'"):
        ws.allocate_process_diagnostics("ocean", {"surface_temperature": _meta(dims.CellDim)})


def test_update_prognostics_syncs_the_written_fields():
    grid = simple.simple_grid()
    exchange = _RecordingExchange()
    ws = _physics_state(grid, exchange)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    ws.compute_diagnostics(prognostic, tracers, _diagnostic_state(grid))
    acc = _accumulated(
        grid, ws, tend_temperature=data_alloc.constant_field(grid, 1e-3, dims.CellDim, dims.KDim)
    )
    perturbed_exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

    ws.update_prognostics(acc.sums, dt_seconds=_DT_SECONDS, perturbed_exner=perturbed_exner)

    # identity, not equality: the sync must target the very buffers that were written;
    # no wind tendency was produced, so vn was neither written nor synced
    assert exchange.events == [
        (
            dims.CellDim,
            [
                *(id(getattr(tracers, name)) for name in physics_state.MOISTURE_SPECIES),
                id(prognostic.w),
                id(prognostic.exner),
                id(prognostic.theta_v),
                id(perturbed_exner),
            ],
        ),
    ]

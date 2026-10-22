# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the PhysicsState layer (PhysicsState facade, process outputs, tendency sums, single update)."""

import dataclasses

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims
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


def _physics_state(grid) -> physics_state.PhysicsState:
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
    return physics_state.PhysicsState(
        grid=grid,
        interpolation=interpolation,
        metrics=metrics,
        backend=None,
    )


# ---------------------------------------------------------------------------
# PhysicsState
# ---------------------------------------------------------------------------


def test_diagnose_fills_working_fields_and_leaves_inputs_untouched():
    """The dyn2phy diagnosis fills plausible fields and is strictly read-only.

    The prognostic state and tracers must stay bitwise identical: only
    ``Tendencies.advance`` and ``Tendencies.update_prognostics`` change them.
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
    assert ws.advanced.temperature is ws.diagnostics.temperature
    assert ws.advanced.u is ws.diagnostics.u
    assert ws.advanced.v is ws.diagnostics.v
    assert ws.advanced.w is prognostic.w
    for name in physics_state.MOISTURE_SPECIES:
        assert getattr(ws.advanced, name) is getattr(tracers, name)


# ---------------------------------------------------------------------------
# Summing the tendencies
# ---------------------------------------------------------------------------


def test_advance_sums_tendencies_and_skips_diagnostics():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), _diagnostic_state(grid))
    acc = _tendencies(grid)
    props = {"tend_qv": _meta(kind=_TENDENCY), "km": _meta()}
    out = {
        "tend_qv": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim),
        "km": data_alloc.constant_field(grid, 5.0, dims.CellDim, dims.KDim),
    }

    acc.zero()
    acc.advance(ws, out, props, _DT_SECONDS)
    acc.advance(ws, out, props, _DT_SECONDS)  # a second process contributing the same tendency

    np.testing.assert_allclose(acc.acc["tend_qv"].asnumpy(), 2e-7, rtol=1e-12)
    assert "km" not in acc.acc


def test_zero_resets_between_steps():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), _diagnostic_state(grid))
    acc = _tendencies(grid)
    props = {"tend_qv": _meta(kind=_TENDENCY)}
    out = {"tend_qv": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim)}

    acc.zero()
    acc.advance(ws, out, props, _DT_SECONDS)
    acc.zero()
    acc.advance(ws, out, props, _DT_SECONDS)

    np.testing.assert_allclose(acc.acc["tend_qv"].asnumpy(), 1e-7, rtol=1e-12)


# ---------------------------------------------------------------------------
# Updating the state with the tendencies
# ---------------------------------------------------------------------------


def _tendencies(grid) -> physics_state.Tendencies:
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
    return physics_state.Tendencies(
        grid=grid, geometry=geometry, interpolation=interpolation, backend=None
    )


def _accumulated(grid, state, **tendencies) -> physics_state.Tendencies:
    """Tendencies of one process with the given constant tendencies, advanced into ``state``."""
    acc = _tendencies(grid)
    props = {name: _meta(kind=_TENDENCY) for name in tendencies}
    acc.zero()
    acc.advance(state, tendencies, props, _DT_SECONDS)
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

    acc.update_prognostics(ws, dt_seconds=dt)

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
    ws = _physics_state(grid)
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

    acc.update_prognostics(ws, dt_seconds=dt)

    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * dt, rtol=1e-12)


def test_tendencies_start_with_a_zero_sum_per_advanced_field():
    tendencies = _tendencies(simple.simple_grid())

    assert set(tendencies.acc) == {
        physics_state.tendency_name(field.name)
        for field in dataclasses.fields(physics_state.AdvancedState)
    }
    for buffer in tendencies.acc.values():
        np.testing.assert_array_equal(buffer.asnumpy(), 0.0)
    # the dims come from the field annotations: w lives on half levels
    assert tendencies.acc["tend_w"].domain.dims == (dims.CellDim, dims.KHalfDim)
    assert tendencies.acc["tend_temperature"].domain.dims == (dims.CellDim, dims.KDim)


def test_advance_rejects_a_tendency_of_no_advanced_field():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), _diagnostic_state(grid))
    acc = _tendencies(grid)
    out = {"tend_dummy": data_alloc.constant_field(grid, 1e-7, dims.CellDim, dims.KDim)}

    acc.zero()
    with pytest.raises(ValueError, match="'tend_dummy' is not the tendency"):
        acc.advance(ws, out, {"tend_dummy": _meta(kind=_TENDENCY)}, _DT_SECONDS)


def test_physics_state_links_the_callers_diagnostic_state():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    diagnostics = _diagnostic_state(grid)
    temperature = diagnostics.temperature

    ws.compute_diagnostics(_uniform_prognostic(grid), _tracer_state(grid), diagnostics)

    # linked, not copied: the caller's fields are the ones diagnosed and advanced
    assert ws.diagnostics is diagnostics
    assert ws.advanced.temperature is temperature
    assert 200.0 < temperature.asnumpy().mean() < 320.0
    # the flat shorthand is gone -- pointers and diagnostics are structurally distinct
    assert not hasattr(ws, "ta")
    assert not hasattr(ws, "pressure")


# ---------------------------------------------------------------------------
# Process outputs
# ---------------------------------------------------------------------------


def test_allocate_process_outputs_from_metadata():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    props = {
        "tend_temperature": _meta(dims.CellDim, dims.KDim, kind=_TENDENCY),
        "kh": _meta(dims.CellDim, dims.KDim),
        "cptgz_vi": _meta(dims.CellDim),
    }
    buffers = ws.allocate_process_outputs("tmx", props)
    # tendencies are never layer-allocated; diagnostics get their declared shape
    assert set(buffers) == {"kh", "cptgz_vi"}
    assert buffers["kh"].domain.dims == (dims.CellDim, dims.KDim)
    assert buffers["cptgz_vi"].domain.dims == (dims.CellDim,)
    assert ws.process_outputs["tmx"] is buffers


def test_allocate_process_outputs_of_a_half_level_output_from_khalfdim():
    # Half levels are stated by the dimension alone, so a KHalfDim output must get
    # one level more than a full-level one.
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    buffers = ws.allocate_process_outputs("tmx", {"kh": _meta(dims.CellDim, dims.KHalfDim)})

    assert buffers["kh"].domain.dims == (dims.CellDim, dims.KHalfDim)
    assert buffers["kh"].ndarray.shape == (grid.num_cells, grid.num_levels + 1)


def test_allocate_process_outputs_keeps_processes_apart():
    # Keyed per process: two processes may declare the same output name.
    # TODO(Chia Rui): Maybe this is not always the case.
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    props = {"kh": _meta(dims.CellDim, dims.KDim)}

    ws.allocate_process_outputs("muphys", props)
    ws.allocate_process_outputs("tmx", props)

    assert ws.process_outputs["muphys"]["kh"] is not ws.process_outputs["tmx"]["kh"]


def test_allocate_process_outputs_rejects_a_duplicate_process_name():
    # A second allocation under the same name would silently orphan the buffers
    # already bound to the first process's component.
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    props = {"kh": _meta(dims.CellDim, dims.KDim)}
    ws.allocate_process_outputs("tmx", props)

    with pytest.raises(ValueError, match="already has output buffers"):
        ws.allocate_process_outputs("tmx", props)


def test_allocate_process_outputs_rejects_a_diagnostic_without_dims():
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    with pytest.raises(ValueError, match="declares no 'dims'"):
        ws.allocate_process_outputs("tmx", {"kh": _meta()})


def test_process_outputs_are_not_set_as_attributes():
    # The outputs stay behind 'process_outputs', so an output name can never shadow
    # a model-state pointer or the diagnostics.
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    ws.allocate_process_outputs("tmx", {"kh": _meta(dims.CellDim, dims.KDim)})

    assert not hasattr(ws, "kh")


# ---------------------------------------------------------------------------
# The surface state
# ---------------------------------------------------------------------------


def test_surface_state_is_allocated_with_the_physics_state():
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    fields = ws.surface.fields()
    assert set(fields) == {
        "evapotranspiration",
        "sensible_heat_flux",
        "latent_heat_flux",
        "u_stress",
        "v_stress",
        "q_snocpymlt",
        "thermal_energy_flux",
        "vapor_energy_flux",
        "exchange_coefficient_momentum",
        "exchange_coefficient_heat",
        "surface_temperature",
    }
    for field in fields.values():
        assert field.domain.dims == (dims.CellDim,)
        assert np.all(field.asnumpy() == 0.0)


def test_allocate_process_outputs_binds_surface_fields_and_allocates_the_rest():
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    buffers = ws.allocate_process_outputs(
        "surface",
        {"sensible_heat_flux": _meta(dims.CellDim), "kh": _meta(dims.CellDim, dims.KDim)},
    )

    # identity: the surface process writes the shared surface state in place
    assert buffers["sensible_heat_flux"] is ws.surface.fluxes.sensible_heat_flux
    assert buffers["kh"] is not ws.surface.fluxes.sensible_heat_flux
    assert buffers["kh"].domain.dims == (dims.CellDim, dims.KDim)


def test_allocate_process_outputs_rejects_a_second_writer_of_a_surface_field():
    grid = simple.simple_grid()
    ws = _physics_state(grid)
    ws.allocate_process_outputs("surface", {"surface_temperature": _meta(dims.CellDim)})

    with pytest.raises(ValueError, match="already written by process 'surface'"):
        ws.allocate_process_outputs("ocean", {"surface_temperature": _meta(dims.CellDim)})


def test_allocate_process_outputs_rejects_wrong_dims_for_a_surface_field():
    grid = simple.simple_grid()
    ws = _physics_state(grid)

    with pytest.raises(ValueError, match="declares dims"):
        ws.allocate_process_outputs(
            "surface", {"sensible_heat_flux": _meta(dims.CellDim, dims.KDim)}
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for implementations/warm_loop.py (M2a Task 1): the warm-phase
operator-split host-loop skeleton, per
docs/superpowers/facts/m2/micro-tendency-orchestration.md ("G1" below).

Groups, matching the task brief's test list:
* TestSubstepDts -- dt_cl/dt_vp exact (G1 §3).
* TestRefreshStateCallCount -- substep counts (n_step_cl x n_step_vp) drive
  the right number of `_refresh_state` calls.
* TestActivation -- `_activation` (M2a Task 4, DONE): `state.diag`
  precondition + no-water no-op.
* TestVaporDepositionLiquid -- `_vapor_deposition_liquid` (M2a Task 5,
  DONE): `state.diag` precondition + no-water no-op.
* TestRepair -- `_repair` (M2a Task 6, DONE): dispatches `phase="collision"`
  to `core.repair.repair_liquid` (af_col) and `phase="vapor"` to
  `core.repair.repair_vapor` (af_vap) -- these are DIFFERENT algorithms
  (code-review-caught: an earlier draft called `repair_liquid` for both,
  incorrectly clipping concentration on every vapor substep); no
  `state.diag` precondition, each phase leaves the other phase's own
  fields untouched.
* TestRefreshStateDiagT -- `_refresh_state` reproduces `core.thermo.diag_t`
  (F1 §5) on a known state.
* TestWarmLoopStateValidation -- npoints-consistency guard.
* TestIfcWarm -- end-to-end wiring smoke test (mocked AND, since M2a Task 6,
  fully UN-mocked over real `bin_grid.LIQUID_NBINS`-sized bins).
* TestEndToEndSupersaturatedSpinUp -- M2a Task 6's own explicit "Tests"
  requirement: a synthetic multi-bin supersaturated spin-up-like state runs
  `run_warm_micro_tendency` end-to-end (all three process hooks real) and
  produces a finite, non-negative liquid state.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps, thermo
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import (
    ScaleRawState,
    get_thermo_prop,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.implementations import warm_loop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    IceState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def luts() -> AmpsLuts:
    return load_luts()


def _thermo_state(
    *, ptotv: float, tv: float, qvv: float = 1.0e-2, moist_denv: float = 1.2e-3
) -> ThermoState:
    """Single-point ThermoState; only ptotv/tv/qvv/moist_denv are given
    physically meaningful values (the rest default to `tv`/0, unused by
    anything under test here)."""
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: ptotv,
        ThermoProp.tv: tv,
        ThermoProp.thv: tv,
        ThermoProp.piv: 0.0,
        ThermoProp.pbv: 0.0,
        ThermoProp.moist_denv: moist_denv,
        ThermoProp.qvv: qvv,
        ThermoProp.thetav: tv,
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]
    return ThermoState(values=values)


def _liquid_state_one_bin(*, rmt: float, rmat: float = 0.0) -> LiquidState:
    """Single-point, single-bin LiquidState with only rmt_q/rmat_q set."""
    lp = index_maps.LiquidPPV
    values = np.zeros((len(LiquidState.PROPS), 1, 1, 1), dtype=np.float64)
    values[lp.rmt_q.py_idx, 0, 0, 0] = rmt
    values[lp.rmat_q.py_idx, 0, 0, 0] = rmat
    return LiquidState(values=values)


def _zero_aerosol_state(npoints: int = 1) -> AerosolState:
    return AerosolState(values=np.zeros((len(AerosolState.PROPS), 1, 1, npoints), dtype=np.float64))


def _make_warm_state(
    *,
    ptotv: float = float(AmpsConst.p00),
    tv: float = 260.0,
    thil: float = 260.0,
    qtp: float = 1.0e-2,
    rmt: float = 0.0,
    rmat: float = 0.0,
    qvv: float = 1.0e-2,
) -> warm_loop.WarmLoopState:
    return warm_loop.WarmLoopState(
        thermo=_thermo_state(ptotv=ptotv, tv=tv, qvv=qvv),
        liquid=_liquid_state_one_bin(rmt=rmt, rmat=rmat),
        aerosol=_zero_aerosol_state(),
        thil=np.array([thil], dtype=np.float64),
        qtp=np.array([qtp], dtype=np.float64),
        mes_rc=np.zeros(1, dtype=np.int64),
    )


def _no_op_process_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _repair/_activation/_vapor_deposition_liquid to pass-throughs
    so run_warm_micro_tendency can run to completion (used by tests that
    are probing the LOOP STRUCTURE, not the stubs' own raise behavior)."""
    monkeypatch.setattr(warm_loop, "_repair", lambda s, c, phase: s)
    monkeypatch.setattr(warm_loop, "_activation", lambda s, c, dt, luts_: s)
    monkeypatch.setattr(warm_loop, "_vapor_deposition_liquid", lambda s, c, dt, luts_: s)


# ---------------------------------------------------------------------------
# dt_cl / dt_vp -- G1 §3, exact.
# ---------------------------------------------------------------------------


class TestSubstepDts:
    def test_cloudlab_values(self):
        config = AmpsConfig.cloudlab()  # n_step_cl=1, n_step_vp=10
        dts = warm_loop._substep_dts(1.0, config)
        assert dts.dt_cl == 1.0 / 1
        assert dts.dt_vp == (1.0 / 1) / 10

    def test_exact_for_arbitrary_dt_and_counts(self):
        config = dataclasses.replace(AmpsConfig.cloudlab(), n_step_cl=4, n_step_vp=3)
        dt = 7.5
        dts = warm_loop._substep_dts(dt, config)
        assert dts.dt_cl == dt / 4
        assert dts.dt_vp == (dt / 4) / 3

    def test_run_warm_micro_tendency_passes_exact_dt_vp_to_process_stubs(self, monkeypatch, luts):
        config = AmpsConfig.cloudlab()
        monkeypatch.setattr(warm_loop, "_repair", lambda s, c, phase: s)
        monkeypatch.setattr(warm_loop, "_vapor_deposition_liquid", lambda s, c, dt, luts_: s)
        seen_dts = []

        def _capture_activation(s, c, dt, luts_):
            seen_dts.append(dt)
            return s

        monkeypatch.setattr(warm_loop, "_activation", _capture_activation)

        dt = 3.0
        expected_dt_vp = (dt / config.n_step_cl) / config.n_step_vp
        warm_loop.run_warm_micro_tendency(_make_warm_state(), config, dt, luts)

        assert len(seen_dts) == config.n_step_cl * config.n_step_vp
        assert all(seen == expected_dt_vp for seen in seen_dts)


# ---------------------------------------------------------------------------
# Substep counts drive the right number of _refresh_state calls.
# ---------------------------------------------------------------------------


class TestRefreshStateCallCount:
    def _run_and_count(self, monkeypatch, config, luts, dt=1.0):
        _no_op_process_stubs(monkeypatch)
        calls = {"n": 0}

        def _counting_refresh(s, config=None, luts=None):
            calls["n"] += 1
            return s

        monkeypatch.setattr(warm_loop, "_refresh_state", _counting_refresh)
        warm_loop.run_warm_micro_tendency(_make_warm_state(), config, dt, luts)
        return calls["n"]

    def test_cloudlab_n_step_cl_1_n_step_vp_10(self, monkeypatch, luts):
        config = AmpsConfig.cloudlab()
        assert config.n_step_cl == 1
        assert config.n_step_vp == 10
        n_calls = self._run_and_count(monkeypatch, config, luts)
        # (n_step_cl - 1) [col-loop it_cl>1, conditional] + n_step_cl*n_step_vp
        # [vap-loop head, unconditional] + 1 [final post-loop refresh].
        assert n_calls == (1 - 1) + 1 * 10 + 1
        assert n_calls == 11

    def test_n_step_cl_3_n_step_vp_2(self, monkeypatch, luts):
        config = dataclasses.replace(AmpsConfig.cloudlab(), n_step_cl=3, n_step_vp=2)
        n_calls = self._run_and_count(monkeypatch, config, luts)
        assert n_calls == (3 - 1) + 3 * 2 + 1
        assert n_calls == 9

    def test_n_step_cl_1_n_step_vp_1(self, monkeypatch, luts):
        config = dataclasses.replace(AmpsConfig.cloudlab(), n_step_cl=1, n_step_vp=1)
        n_calls = self._run_and_count(monkeypatch, config, luts)
        assert n_calls == (1 - 1) + 1 * 1 + 1
        assert n_calls == 2


# ---------------------------------------------------------------------------
# _activation (M2a Task 4, DONE): no longer a stub -- requires state.diag
# (populated by _refresh_state), then delegates to
# core.activation.activate_and_advance_vapor.
# ---------------------------------------------------------------------------


class TestActivation:
    def test_raises_without_diag(self, luts):
        """Calling `_activation` directly on a state that skipped
        `_refresh_state` (`diag=None`, `WarmLoopState`'s own default) must
        raise a clear `ValueError`, not silently misbehave."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state()
        assert state.diag is None
        with pytest.raises(ValueError, match="state\\.diag"):
            warm_loop._activation(state, config, 0.01, luts)

    def test_no_water_box_is_a_no_op_after_refresh(self, luts):
        """A `diag`-populated, no-water box (mes_rc==0): CCN activation's
        own grid-box skip mask (G2 section 1a) leaves liquid/aerosol/thermo
        unchanged."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(qtp=0.0, rmt=0.0, rmat=0.0, qvv=0.0)
        refreshed = warm_loop._refresh_state(state, config, luts)
        assert refreshed.diag is not None

        activated = warm_loop._activation(refreshed, config, 0.01, luts)
        np.testing.assert_array_equal(activated.liquid.values, refreshed.liquid.values)
        np.testing.assert_array_equal(activated.aerosol.values, refreshed.aerosol.values)
        np.testing.assert_array_equal(activated.thermo.values, refreshed.thermo.values)


# ---------------------------------------------------------------------------
# _vapor_deposition_liquid (M2a Task 5, DONE): no longer a stub -- requires
# state.diag (populated by _refresh_state), then delegates to
# core.vapor_deposition.vapor_deposition_liquid.
# ---------------------------------------------------------------------------


class TestVaporDepositionLiquid:
    def test_raises_without_diag(self, luts):
        """Calling `_vapor_deposition_liquid` directly on a state that
        skipped `_refresh_state` (`diag=None`, `WarmLoopState`'s own
        default) must raise a clear `ValueError`, not silently misbehave
        -- matching `_activation`'s own precondition check."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state()
        assert state.diag is None
        with pytest.raises(ValueError, match="state\\.diag"):
            warm_loop._vapor_deposition_liquid(state, config, 0.01, luts)

    def test_no_water_box_is_a_no_op_after_refresh(self, luts):
        """A `diag`-populated, no-water box: no active bins (`diag.
        mean_mass<=0` everywhere) -> every bin is a passthrough ->
        liquid unchanged."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(qtp=0.0, rmt=0.0, rmat=0.0, qvv=0.0)
        refreshed = warm_loop._refresh_state(state, config, luts)
        assert refreshed.diag is not None

        deposited = warm_loop._vapor_deposition_liquid(refreshed, config, 0.01, luts)
        np.testing.assert_array_equal(deposited.liquid.values, refreshed.liquid.values)


# ---------------------------------------------------------------------------
# _repair (M2a Task 6, DONE): no longer a stub -- dispatches phase=
# "collision" to core.repair.repair_liquid (af_col, per-bin mass/
# concentration non-negativity) and phase="vapor" to core.repair.
# repair_vapor (af_vap, point-level vapor-supply closure). These are
# DIFFERENT algorithms in G5 -- see warm_loop.py's own _repair docstring
# and core/repair.py's module docstring. A code-review-caught earlier
# draft of this wiring incorrectly called repair_liquid for both phases.
# ---------------------------------------------------------------------------


class TestRepair:
    def test_collision_phase_clips_negative_mass_bin(self):
        """`phase="collision"` delegates to `core.repair.repair_liquid` --
        a state with a small negative bin mass comes back non-negative,
        matching `test_repair.py`'s own `TestRepairLiquid` at the
        `WarmLoopState` integration level."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=-1.0e-8)

        repaired = warm_loop._repair(state, config, phase="collision")

        lp = index_maps.LiquidPPV
        assert repaired.liquid.values[lp.rmt_q.py_idx, 0, 0, 0] == 0.0

    def test_repair_does_not_require_diag(self):
        """Unlike `_activation`/`_vapor_deposition_liquid`, `_repair`
        operates directly on `state.liquid`/`state.thermo` -- it has no
        `state.diag` precondition (matching `run_warm_micro_tendency`'s
        own G1 §1 wiring: the FIRST col-loop `_repair` call happens BEFORE
        any refresh, `it_cl>1` being false on the first iteration)."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=-1.0e-8)
        assert state.diag is None

        repaired = warm_loop._repair(state, config, phase="collision")

        lp = index_maps.LiquidPPV
        assert repaired.liquid.values[lp.rmt_q.py_idx, 0, 0, 0] == 0.0

    def test_vapor_phase_enforces_qvv_nonnegativity(self):
        """`phase="vapor"` delegates to `core.repair.repair_vapor` (af_vap)
        -- a state with negative `thermo.qvv` (over-condensation this
        substep) comes back with `qvv>=0`, matching `test_repair.py`'s own
        `TestRepairVapor` at the `WarmLoopState` integration level."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=5.0e-6, qvv=-1.0e-6)

        repaired = warm_loop._repair(state, config, phase="vapor")

        qvv_after = get_thermo_prop(repaired.thermo, ThermoProp.qvv)[0]
        assert qvv_after == pytest.approx(0.0, abs=1e-20)

    def test_vapor_phase_never_touches_concentration(self):
        """THE property the code review flagged as broken: af_vap's real
        Fortran algorithm (`cal_mass_budget_vapor`) has no `dcondt`
        anywhere in its body -- `rcon_q` must be bit-for-bit unchanged
        after `phase="vapor"`, unlike `phase="collision"` (which DOES clip
        `rcon_q`, see `TestRepairLiquid`)."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=5.0e-6, qvv=-1.0e-6)

        repaired = warm_loop._repair(state, config, phase="vapor")

        lp = index_maps.LiquidPPV
        np.testing.assert_array_equal(
            repaired.liquid.values[lp.rcon_q.py_idx], state.liquid.values[lp.rcon_q.py_idx]
        )

    def test_collision_and_vapor_phases_are_different_algorithms(self):
        """Negative regression for the code-review bug (an earlier draft
        called `repair_liquid` for both phases). Given a state with
        POSITIVE bin mass (already non-negative -- outside af_col's own
        trigger) but negative `qvv` (a vapor-supply problem, outside
        af_col's scope entirely): collision phase is a complete no-op;
        vapor phase actively reduces liquid mass and repairs `qvv`. The
        two phases must genuinely diverge, not just differ by
        happenstance."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=5.0e-6, qvv=-1.0e-6)

        collision = warm_loop._repair(state, config, phase="collision")
        vapor = warm_loop._repair(state, config, phase="vapor")

        lp = index_maps.LiquidPPV
        # collision: no-op (mass/con already non-negative, qvv out of scope).
        np.testing.assert_array_equal(collision.liquid.values, state.liquid.values)
        np.testing.assert_array_equal(collision.thermo.values, state.thermo.values)
        # vapor: mass reduced to repay the deficit, qvv repaired to 0.
        assert (
            vapor.liquid.values[lp.rmt_q.py_idx, 0, 0, 0]
            < state.liquid.values[lp.rmt_q.py_idx, 0, 0, 0]
        )
        qvv_after = get_thermo_prop(vapor.thermo, ThermoProp.qvv)[0]
        assert qvv_after == pytest.approx(0.0, abs=1e-20)
        # The two phases therefore produce genuinely different output.
        assert not np.array_equal(collision.liquid.values, vapor.liquid.values)

    def test_repair_leaves_other_fields_untouched(self):
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=-1.0e-8)

        repaired = warm_loop._repair(state, config, phase="collision")

        np.testing.assert_array_equal(repaired.thermo.values, state.thermo.values)
        np.testing.assert_array_equal(repaired.aerosol.values, state.aerosol.values)
        np.testing.assert_array_equal(repaired.thil, state.thil)
        np.testing.assert_array_equal(repaired.qtp, state.qtp)

    def test_vapor_phase_leaves_aerosol_and_column_scalars_untouched(self):
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(rmt=5.0e-6, qvv=-1.0e-6)

        repaired = warm_loop._repair(state, config, phase="vapor")

        np.testing.assert_array_equal(repaired.aerosol.values, state.aerosol.values)
        np.testing.assert_array_equal(repaired.thil, state.thil)
        np.testing.assert_array_equal(repaired.qtp, state.qtp)

    def test_invalid_phase_raises(self):
        config = AmpsConfig.cloudlab()
        state = _make_warm_state()
        with pytest.raises(ValueError, match="phase"):
            warm_loop._repair(state, config, phase="bogus")

    def test_run_warm_micro_tendency_no_longer_raises_notimplementederror(self, luts):
        """cloudlab (n_step_cl=1): the FIRST col-loop iteration calls
        _repair unconditionally (G1 §1, before any refresh), which used to
        raise (Task 1-5 stub) -- now that Task 6 implements it, a truly
        no-water state (`qtp=rmt=rmat=qvv=0`, `TestActivation`'s/
        `TestVaporDepositionLiquid`'s own no-op precedent -- `_activation`/
        `_vapor_deposition_liquid` skip before ever needing a real
        `bin_grid.LIQUID_NBINS`-sized liquid state, so the single-bin
        `_make_warm_state()` fixture is safe here) runs to completion
        without raising and returns a finite state."""
        config = AmpsConfig.cloudlab()
        state = _make_warm_state(qtp=0.0, rmt=0.0, rmat=0.0, qvv=0.0)
        result = warm_loop.run_warm_micro_tendency(state, config, 1.0, luts)
        assert np.all(np.isfinite(result.liquid.values))
        assert np.all(np.isfinite(result.thermo.values))


# ---------------------------------------------------------------------------
# _refresh_state reproduces core.thermo.diag_t (F1 §5) on a known state.
# ---------------------------------------------------------------------------


class TestRefreshStateDiagT:
    def test_matches_diag_t_quadratic_branch_with_rain(self):
        """P == p00 removes the Exner term (til == thil); rmt-rmat gives a
        known qr contributing latent heat, taking diag_t's quadratic
        branch (T >= 253 K) -- mirrors test_thermo.py's own round-trip
        cases but through _refresh_state's full mes_rc -> qr -> diag_t
        pipeline instead of calling thermo.diag_t directly."""
        p = float(AmpsConst.p00)
        thil_val = 260.0
        rmt, rmat = 3.0e-3, 5.0e-4
        state = _make_warm_state(ptotv=p, thil=thil_val, rmt=rmt, rmat=rmat)

        refreshed = warm_loop._refresh_state(state)

        qr_expected = max(0.0, rmt - rmat)
        t_expected, ierror_expected = thermo.diag_t(thil_val, p, qr_expected, 0.0)
        t_actual = get_thermo_prop(refreshed.thermo, ThermoProp.tv)[0]

        assert t_actual == pytest.approx(t_expected, rel=1e-13)
        assert ierror_expected == 0
        assert refreshed.mes_rc[0] == 2  # rain present

    def test_matches_diag_t_linear_branch_no_rain(self):
        """rmt=rmat=0 -> no rain mass -> mes_rc=1 (vapor only) -> qr=0 ->
        diag_t's linear branch (til == T directly, T < 253)."""
        p = float(AmpsConst.p00)
        thil_val = 240.0
        state = _make_warm_state(ptotv=p, thil=thil_val, rmt=0.0, rmat=0.0)

        refreshed = warm_loop._refresh_state(state)

        t_expected, ierror_expected = thermo.diag_t(thil_val, p, 0.0, 0.0)
        t_actual = get_thermo_prop(refreshed.thermo, ThermoProp.tv)[0]

        assert t_actual == pytest.approx(t_expected, rel=1e-13)
        assert ierror_expected == 0
        assert refreshed.mes_rc[0] == 1  # vapor only, no rain

    def test_no_water_gives_mes_rc_zero(self):
        """qvv=0 and rmt=0 -> M_tot<=0 -> mes_rc=0 (no water), G1 §4a."""
        state = _make_warm_state(rmt=0.0, rmat=0.0, qvv=0.0)
        refreshed = warm_loop._refresh_state(state)
        assert refreshed.mes_rc[0] == 0

    def test_thil_and_qtp_held_fixed_across_refresh(self):
        """G1 §1: thil(*)/qtp(*) are plain incoming arguments, never
        reassigned by the refresh preamble -- only T is re-diagnosed."""
        state = _make_warm_state(thil=255.0, qtp=0.0123)
        refreshed = warm_loop._refresh_state(state)
        assert refreshed.thil == pytest.approx(state.thil)
        assert refreshed.qtp == pytest.approx(state.qtp)

    def test_vectorized_matches_per_point_diag_t(self):
        """Multi-point WarmLoopState: each point's diag_t inversion must be
        independent (no cross-point leakage in the mes_rc/qr reduction)."""
        p = float(AmpsConst.p00)
        thil_vals = np.array([240.0, 260.0, 265.0])
        rmt_vals = np.array([0.0, 2.0e-3, 5.0e-3])
        rmat_vals = np.array([0.0, 1.0e-4, 1.0e-3])

        lp = index_maps.LiquidPPV
        liquid_values = np.zeros((len(LiquidState.PROPS), 1, 1, 3))
        liquid_values[lp.rmt_q.py_idx, 0, 0, :] = rmt_vals
        liquid_values[lp.rmat_q.py_idx, 0, 0, :] = rmat_vals

        state = warm_loop.WarmLoopState(
            thermo=_thermo_state_multi(ptotv=p, tv=thil_vals, npoints=3),
            liquid=LiquidState(values=liquid_values),
            aerosol=_zero_aerosol_state(npoints=3),
            thil=thil_vals.copy(),
            qtp=np.full(3, 1.0e-2),
            mes_rc=np.zeros(3, dtype=np.int64),
        )

        refreshed = warm_loop._refresh_state(state)
        t_actual = get_thermo_prop(refreshed.thermo, ThermoProp.tv)

        qr_expected = np.maximum(0.0, rmt_vals - rmat_vals)
        t_expected, _ = thermo.diag_t(thil_vals, p, qr_expected, 0.0)
        np.testing.assert_allclose(t_actual, t_expected, rtol=1e-13)


def _thermo_state_multi(*, ptotv: float, tv: np.ndarray, npoints: int) -> ThermoState:
    values = np.zeros((len(ThermoState.PROPS), 1, 1, npoints), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: np.full(npoints, ptotv),
        ThermoProp.tv: tv,
        ThermoProp.thv: tv,
        ThermoProp.piv: np.zeros(npoints),
        ThermoProp.pbv: np.zeros(npoints),
        ThermoProp.moist_denv: np.full(npoints, 1.2e-3),
        ThermoProp.qvv: np.full(npoints, 1.0e-2),
        ThermoProp.thetav: tv,
        ThermoProp.wbv: np.zeros(npoints),
        ThermoProp.momv: np.zeros(npoints),
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        values[idx, 0, 0, :] = by_prop[ThermoProp(int(prop))]
    return ThermoState(values=values)


# ---------------------------------------------------------------------------
# WarmLoopState -- npoints-consistency validation.
# ---------------------------------------------------------------------------


class TestWarmLoopStateValidation:
    def test_mismatched_npoints_raises(self):
        with pytest.raises(ValueError, match="npoints"):
            warm_loop.WarmLoopState(
                thermo=_thermo_state(ptotv=float(AmpsConst.p00), tv=260.0),
                liquid=_liquid_state_one_bin(rmt=0.0),
                aerosol=_zero_aerosol_state(npoints=2),  # mismatched
                thil=np.array([260.0]),
                qtp=np.array([1.0e-2]),
                mes_rc=np.zeros(1, dtype=np.int64),
            )

    def test_consistent_npoints_constructs_ok(self):
        state = _make_warm_state()
        assert state.thermo.npoints == 1


# ---------------------------------------------------------------------------
# ifc_warm -- end-to-end wiring smoke test (process stubs mocked out).
# ---------------------------------------------------------------------------


class TestIfcWarm:
    # NBR (liquid bins) must be a valid bin_grid.LIQUID_NBINS count (40 or
    # 80) for the UNMOCKED end-to-end test below, which exercises real
    # `core.activation`/`core.vapor_deposition` physics that call
    # `bin_grid.make_bin_grid("liquid", ...)`; the mocked test doesn't care
    # (process stubs bypass real physics entirely) so this enlargement is
    # safe for both.
    NBR = 40
    NBI = 2
    NBA = 2
    NPOINTS = 2

    def _make_scale_raw(self) -> ScaleRawState:
        rng = np.random.default_rng(0)
        npoints = self.NPOINTS
        dens = rng.uniform(0.9, 1.1, size=npoints)
        qdry = rng.uniform(0.95, 0.999, size=npoints)
        qv = rng.uniform(1.0e-4, 5.0e-3, size=npoints)
        pres = np.full(npoints, float(AmpsConst.p00) * 0.1)  # CGS p00 -> Pa
        temp = rng.uniform(260.0, 285.0, size=npoints)
        w = np.zeros(npoints)
        momz = np.zeros(npoints)
        ql = rng.uniform(1.0e-6, 1.0e-4, size=(self.NBR, npoints))
        qi = np.zeros((self.NBI, npoints))

        liquid_raw = LiquidState(
            values=rng.uniform(1.0e-7, 1.0e-5, size=(len(LiquidState.PROPS), self.NBR, 1, npoints))
        )
        ice_raw = IceState(values=np.zeros((len(IceState.PROPS), self.NBI, 1, npoints)))
        aerosol_raw = AerosolState(
            values=rng.uniform(1.0e-8, 1.0e-6, size=(len(AerosolState.PROPS), self.NBA, 1, npoints))
        )

        return ScaleRawState(
            dens=dens,
            qdry=qdry,
            qv=qv,
            pres=pres,
            temp=temp,
            w=w,
            momz=momz,
            ql=ql,
            qi=qi,
            liquid_ppv=liquid_raw,
            ice_ppv=ice_raw,
            aerosol_ppv=aerosol_raw,
        )

    def test_runs_end_to_end_with_process_stubs_mocked(self, monkeypatch, luts):
        _no_op_process_stubs(monkeypatch)
        monkeypatch.setattr(warm_loop, "_refresh_state", lambda s, config=None, luts=None: s)

        scale = self._make_scale_raw()
        config = AmpsConfig.cloudlab()
        thil = np.full(self.NPOINTS, 270.0)
        qtp = np.full(self.NPOINTS, 1.0e-2)
        dens_t = np.zeros(self.NPOINTS)

        tendencies = warm_loop.ifc_warm(scale, thil, qtp, config, dt=1.0, luts=luts, dens_t=dens_t)

        assert tendencies.dqv.shape == (self.NPOINTS,)
        assert tendencies.dql.shape == (self.NBR, self.NPOINTS)
        assert np.all(np.isfinite(tendencies.dqv))

    def test_runs_end_to_end_without_mocking_process_stubs(self, luts):
        """M2a Task 6: with all three process hooks (`_activation`,
        `_vapor_deposition_liquid`, `_repair`) now implemented, a full,
        UN-mocked `ifc_warm` run over `NBR=40` real liquid bins (the
        `bin_grid.LIQUID_NBINS` requirement `core.activation`/`core.
        vapor_deposition` enforce) completes without `NotImplementedError`
        and returns finite tendencies -- this superseded the old M2a
        Task 1-5 "raises on first repair" smoke test now that the whole
        warm loop is wired end-to-end."""
        scale = self._make_scale_raw()
        config = AmpsConfig.cloudlab()
        thil = np.full(self.NPOINTS, 270.0)
        qtp = np.full(self.NPOINTS, 1.0e-2)
        dens_t = np.zeros(self.NPOINTS)

        tendencies = warm_loop.ifc_warm(scale, thil, qtp, config, dt=1.0, luts=luts, dens_t=dens_t)

        assert np.all(np.isfinite(tendencies.dqv))
        assert np.all(np.isfinite(tendencies.dql))


# ---------------------------------------------------------------------------
# End-to-end: a synthetic multi-bin supersaturated spin-up-like state runs
# run_warm_micro_tendency with all three M2a Task 4/5/6 process hooks REAL
# (no monkeypatching) -- the task's own explicit "Tests" requirement.
# ---------------------------------------------------------------------------


class TestEndToEndSupersaturatedSpinUp:
    """A cloudlab-shaped (40 liquid bins, `nbin_h=20`), single-column,
    supersaturated, no-pre-existing-liquid state -- a "spin-up-like"
    scenario (droplets nucleate from CCN as the run proceeds), reusing
    `test_activation.py`'s own PROVEN-supersaturated setup
    (`TestMassConservation`: `T_STD=280K`, `P_STD=p00`, `qv=1.15e-2`, CCN
    population `(amt,acon,ams)=(3.164e-13,300.0,3.164e-13)`) -- exercises
    `_activation` -> `_vapor_deposition_liquid` -> `_repair` for real,
    across all `n_step_cl*n_step_vp` substeps."""

    LIQ_NBINS = 40
    NBIN_H = 20

    def _config(self) -> AmpsConfig:
        base = AmpsConfig.cloudlab()
        assert base.num_h_bins[0] == self.LIQ_NBINS
        assert base.nbin_h == self.NBIN_H
        return base

    def _state(self) -> warm_loop.WarmLoopState:
        p = float(AmpsConst.p00)
        t = 280.0
        qv = 1.15e-2  # supersaturated at (p, t) -- test_activation.py's own sanity check

        thermo_values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
        by_prop = {
            ThermoProp.ptotv: p,
            ThermoProp.tv: t,
            ThermoProp.thv: t,
            ThermoProp.piv: 0.0,
            ThermoProp.pbv: 0.0,
            ThermoProp.moist_denv: 1.2e-3,
            ThermoProp.qvv: qv,
            ThermoProp.thetav: t,
            ThermoProp.wbv: 0.0,
            ThermoProp.momv: 0.0,
        }
        for idx, prop in enumerate(ThermoState.PROPS):
            thermo_values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]

        liquid = LiquidState(
            values=np.zeros((len(LiquidState.PROPS), self.LIQ_NBINS, 1, 1), dtype=np.float64)
        )

        aerosol_values = np.zeros((len(AerosolState.PROPS), 1, 1, 1), dtype=np.float64)
        aerosol_values[0, 0, 0, 0] = 3.164e-13  # amt_q
        aerosol_values[1, 0, 0, 0] = 300.0  # acon_q
        aerosol_values[2, 0, 0, 0] = 3.164e-13  # ams_q
        aerosol = AerosolState(values=aerosol_values)

        return warm_loop.WarmLoopState(
            thermo=ThermoState(values=thermo_values),
            liquid=liquid,
            aerosol=aerosol,
            thil=np.array([t]),
            qtp=np.array([qv]),
            mes_rc=np.zeros(1, dtype=np.int64),
        )

    def test_runs_to_completion_and_produces_finite_nonnegative_state(self, luts):
        config = self._config()
        state = self._state()

        result = warm_loop.run_warm_micro_tendency(state, config, dt=1.0, luts=luts)

        assert np.all(np.isfinite(result.liquid.values))
        assert np.all(np.isfinite(result.thermo.values))
        assert np.all(np.isfinite(result.aerosol.values))

        lp = index_maps.LiquidPPV
        assert np.all(result.liquid.values[lp.rmt_q.py_idx] >= 0.0)
        assert np.all(result.liquid.values[lp.rcon_q.py_idx] >= 0.0)

    def test_activation_actually_occurred(self, luts):
        """Sanity: the supersaturated setup is not vacuous -- some liquid
        mass exists after the run (droplets nucleated from CCN, matching
        the "spin-up-like" framing -- otherwise the non-negativity
        assertions above would be trivially true of an all-zero state)."""
        config = self._config()
        state = self._state()

        result = warm_loop.run_warm_micro_tendency(state, config, dt=1.0, luts=luts)

        lp = index_maps.LiquidPPV
        assert result.liquid.values[lp.rmt_q.py_idx].sum() > 0.0

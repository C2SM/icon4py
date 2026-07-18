# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/repair.py (M2a Task 6): `repair`'s TWO independent
liquid-budget-closure phases, per
docs/superpowers/facts/m2/sedimentation-terminalvel.md ("G5") §4 +
`cal_mass_budget_vapor` (`mod_amps_check.F90:2034-3145`, read directly). See
`core/repair.py`'s own module docstring for the full derivation of why both
are state-level closures (with G5's exact `0.99999`/`total_limit` trigger
and `NITER`/`NITER_TOTAL` loop bounds where applicable), not literal
per-process `dmassdt` tendency rescales -- and for why `repair_liquid`
(af_col) and `repair_vapor` (af_vap) are DIFFERENT algorithms, not
interchangeable (a code-review-caught bug in an earlier draft called
`repair_liquid` for both).

Groups:
* TestConstants -- NITER/NITER_TOTAL/TOTAL_LIMIT match G5 exactly.
* TestNonnegRescaleLeg -- the private per-leg convergence loop in isolation.
* TestRepairLiquid -- the public `repair_liquid` (af_col) deliverable:
  negative mass/concentration bins clipped to non-negative, untouched
  elements preserved bit-for-bit, total change bounded by exactly the
  clipped deficit (G5's own degree of conservation -- see module
  docstring's "don't impose stricter conservation than Fortran" note), no
  input mutation, multi-point independence, and the documented
  rmat_q/rmas_q/volume out-of-scope no-ops.
* TestRepairVapor -- the public `repair_vapor` (af_vap) deliverable:
  negative `thermo.qvv` (vapor deficit) repaired to non-negative by taking
  water mass back from `liquid.rmt_q`, concentration (`rcon_q`) NEVER
  touched (the property this class exists to lock in -- af_vap's real
  Fortran algorithm has no `dcondt` in its array list at all), no-op when
  vapor is already non-negative, and the honest "insufficient liquid mass"
  edge case.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import repair
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _liquid_state(*, rmt: np.ndarray, rcon: np.ndarray, rmat=None, rmas=None) -> LiquidState:
    """`(nbins, npoints)` rmt/rcon arrays -> a `LiquidState` (ncat=1).
    rmat_q/rmas_q default to zero unless given (same shape)."""
    nbins, npoints = rmt.shape
    assert rcon.shape == (nbins, npoints)
    lp = LiquidPPV
    values = np.zeros((len(LiquidState.PROPS), nbins, 1, npoints), dtype=np.float64)
    values[lp.rmt_q.py_idx, :, 0, :] = rmt
    values[lp.rcon_q.py_idx, :, 0, :] = rcon
    if rmat is not None:
        values[lp.rmat_q.py_idx, :, 0, :] = rmat
    if rmas is not None:
        values[lp.rmas_q.py_idx, :, 0, :] = rmas
    return LiquidState(values=values)


def _thermo_state(*, qvv: np.ndarray) -> ThermoState:
    """`(npoints,)` qvv array -> a `ThermoState` with every other field
    zero (irrelevant to `repair_vapor`, which reads only `qvv`)."""
    npoints = qvv.shape[0]
    values = np.zeros((len(ThermoState.PROPS), 1, 1, npoints), dtype=np.float64)
    qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
    values[qvv_idx, 0, 0, :] = qvv
    return ThermoState(values=values)


# ---------------------------------------------------------------------------
# TestConstants -- G5 fidelity.
# ---------------------------------------------------------------------------


class TestConstants:
    def test_niter_matches_g5(self):
        """G5 mod_amps_check.F90:112 (`NITER = 300`)."""
        assert repair.NITER == 300

    def test_niter_total_matches_g5(self):
        """G5 mod_amps_check.F90:113 (`NITER_TOTAL = 4`)."""
        assert repair.NITER_TOTAL == 4

    def test_total_limit_matches_g5(self):
        """G5 `cal_mass_budget_col`: `real(PS),parameter :: total_limit=1.0e-30_PS`."""
        assert pytest.approx(1.0e-30) == repair.TOTAL_LIMIT


# ---------------------------------------------------------------------------
# TestNonnegRescaleLeg -- the private per-leg convergence loop.
# ---------------------------------------------------------------------------


class TestNonnegRescaleLeg:
    def test_negative_value_clipped_to_zero(self):
        v = np.array([-1.0e-6])
        out, mark = repair._nonneg_rescale_leg(v)
        assert out[0] == 0.0
        assert mark[0] == 1

    def test_positive_value_unchanged(self):
        v = np.array([3.0e-5])
        out, mark = repair._nonneg_rescale_leg(v)
        assert out[0] == pytest.approx(3.0e-5)
        assert mark[0] == 0

    def test_zero_value_unchanged(self):
        v = np.array([0.0])
        out, mark = repair._nonneg_rescale_leg(v)
        assert out[0] == 0.0
        assert mark[0] == 0

    def test_tiny_negative_below_total_limit_untouched(self):
        """G5's own `total_rloss>total_limit` guard: a loss AT OR BELOW
        1e-30 does not trigger the rescale -- matches Fortran exactly, not
        a bug in this port."""
        v = np.array([-1.0e-32])
        out, mark = repair._nonneg_rescale_leg(v)
        assert out[0] == pytest.approx(-1.0e-32)
        assert mark[0] == 0

    def test_mixed_array_independent_elementwise(self):
        v = np.array([-2.0e-6, 5.0e-6, 0.0, -9.0e-7])
        out, mark = repair._nonneg_rescale_leg(v)
        np.testing.assert_array_equal(out, [0.0, 5.0e-6, 0.0, 0.0])
        np.testing.assert_array_equal(mark, [1, 0, 0, 1])

    def test_does_not_mutate_input_array(self):
        v = np.array([-1.0e-6, 2.0e-6])
        v_before = v.copy()
        repair._nonneg_rescale_leg(v)
        np.testing.assert_array_equal(v, v_before)

    def test_converges_within_niter_bound(self):
        """The derivation (module docstring) shows one pass always
        resolves every trigger; confirm the loop does not run away to
        NITER without exiting early (no RuntimeError, which is the only
        way a non-convergence would surface here)."""
        rng = np.random.default_rng(1)
        v = rng.uniform(-1.0e-4, 1.0e-4, size=1000)
        out, _mark = repair._nonneg_rescale_leg(v)
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# TestRepairLiquid -- the public M2a Task 6 deliverable.
# ---------------------------------------------------------------------------


class TestRepairLiquid:
    def test_negative_mass_bin_repaired_to_nonneg(self):
        """A small negative bin mass (bin 1 of 3) is repaired to
        non-negative; the OTHER bins are conserved bit-for-bit (G5's own
        degree of conservation -- see core/repair.py module docstring)."""
        rmt = np.array([[5.0e-6], [-1.0e-8], [3.0e-6]])
        rcon = np.array([[10.0], [10.0], [10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        lp = LiquidPPV
        new_mass = out.values[lp.rmt_q.py_idx, :, 0, 0]
        assert np.all(new_mass >= 0.0)
        assert new_mass[1] == 0.0
        assert new_mass[0] == pytest.approx(5.0e-6)
        assert new_mass[2] == pytest.approx(3.0e-6)

    def test_negative_concentration_bin_repaired_to_nonneg(self):
        rmt = np.array([[1.0e-6], [1.0e-6]])
        rcon = np.array([[-5.0], [7.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        lp = LiquidPPV
        new_con = out.values[lp.rcon_q.py_idx, :, 0, 0]
        assert new_con[0] == 0.0
        assert new_con[1] == pytest.approx(7.0)

    def test_all_nonnegative_state_is_exact_noop(self):
        rng = np.random.default_rng(2)
        rmt = rng.uniform(1.0e-8, 1.0e-4, size=(5, 3))
        rcon = rng.uniform(0.0, 50.0, size=(5, 3))
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        np.testing.assert_array_equal(out.values, liquid.values)

    def test_total_mass_change_equals_exactly_the_clipped_deficit(self):
        """ "Conserving total to the extent Fortran conserves": the clip
        removes ONLY the negative deficit, no more, no less -- it does not
        redistribute from/to other bins (module docstring: "don't impose
        stricter conservation than Fortran")."""
        deficit = 2.5e-7
        rmt = np.array([[4.0e-6], [-deficit], [6.0e-6]])
        rcon = np.full((3, 1), 10.0)
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        lp = LiquidPPV
        total_before = rmt.sum()
        total_after = out.values[lp.rmt_q.py_idx, :, 0, :].sum()
        assert total_after - total_before == pytest.approx(deficit)

    def test_does_not_mutate_input_liquidstate(self):
        rmt = np.array([[-1.0e-6], [2.0e-6]])
        rcon = np.array([[3.0], [-4.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        values_before = liquid.values.copy()
        config = AmpsConfig.cloudlab()

        repair.repair_liquid(liquid, config)

        np.testing.assert_array_equal(liquid.values, values_before)

    def test_returns_liquidstate_same_shape(self):
        rmt = np.array([[-1.0e-6], [2.0e-6]])
        rcon = np.array([[3.0], [-4.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        assert isinstance(out, LiquidState)
        assert out.values.shape == liquid.values.shape

    def test_tiny_negative_below_total_limit_left_untouched(self):
        """Matches `_nonneg_rescale_leg`'s own G5-fidelity edge case at the
        `repair_liquid` integration level."""
        rmt = np.array([[-1.0e-32], [1.0e-5]])
        rcon = np.full((2, 1), 10.0)
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        lp = LiquidPPV
        assert out.values[lp.rmt_q.py_idx, 0, 0, 0] == pytest.approx(-1.0e-32)

    def test_multipoint_vectorized_independent(self):
        """Two columns (points): a negative bin in point 0 must not affect
        point 1's values."""
        rmt = np.array([[-1.0e-6, 2.0e-6], [3.0e-6, -4.0e-6]])
        rcon = np.full((2, 2), 5.0)
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        lp = LiquidPPV
        new_mass = out.values[lp.rmt_q.py_idx, :, 0, :]
        np.testing.assert_array_equal(new_mass, [[0.0, 2.0e-6], [3.0e-6, 0.0]])

    def test_rmat_rmas_out_of_scope_pass_through_unchanged(self):
        """G5 §4b's quoted block does not address `rmat_q`/`rmas_q`
        (aerosol content) -- documented out of scope; even a negative
        value there passes through untouched (core/repair.py module
        docstring's own scope note)."""
        rmt = np.array([[1.0e-6]])
        rcon = np.array([[10.0]])
        rmat = np.array([[-1.0e-9]])
        rmas = np.array([[-2.0e-9]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon, rmat=rmat, rmas=rmas)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        lp = LiquidPPV
        assert out.values[lp.rmat_q.py_idx, 0, 0, 0] == pytest.approx(-1.0e-9)
        assert out.values[lp.rmas_q.py_idx, 0, 0, 0] == pytest.approx(-2.0e-9)

    def test_finite_result_from_random_state(self):
        rng = np.random.default_rng(3)
        rmt = rng.uniform(-1.0e-6, 1.0e-4, size=(20, 4))
        rcon = rng.uniform(-5.0, 50.0, size=(20, 4))
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        config = AmpsConfig.cloudlab()

        out = repair.repair_liquid(liquid, config)

        assert np.all(np.isfinite(out.values))
        lp = LiquidPPV
        assert np.all(out.values[lp.rmt_q.py_idx] >= 0.0)
        assert np.all(out.values[lp.rcon_q.py_idx] >= 0.0)


# ---------------------------------------------------------------------------
# TestRepairVapor -- the public af_vap (`cal_mass_budget_vapor`) deliverable.
# ---------------------------------------------------------------------------


class TestRepairVapor:
    def test_negative_qvv_repaired_to_nonneg(self):
        """The core af_vap invariant: a vapor deficit (negative `qvv`,
        af_vap's own failure mode) is repaired to non-negative."""
        rmt = np.array([[5.0e-6]])
        rcon = np.array([[10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        thermo_state = _thermo_state(qvv=np.array([-1.0e-6]))
        config = AmpsConfig.cloudlab()

        _liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
        assert thermo_out.values[qvv_idx, 0, 0, 0] == pytest.approx(0.0, abs=1e-20)

    def test_deficit_taken_from_liquid_mass_conserving_total(self):
        """The vapor deficit is given back by removing EXACTLY that much
        water mass from `liquid.rmt_q` -- qvv+liquid mass total is
        conserved (unlike af_col's own repair_liquid, which the module
        docstring explicitly does NOT require to conserve)."""
        rmt = np.array([[5.0e-6]])
        rcon = np.array([[10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        deficit = 8.0e-7
        thermo_state = _thermo_state(qvv=np.array([-deficit]))
        config = AmpsConfig.cloudlab()

        liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
        qvv_after = thermo_out.values[qvv_idx, 0, 0, 0]
        mass_after = liquid_out.values[lp.rmt_q.py_idx, 0, 0, 0]

        assert qvv_after == pytest.approx(0.0, abs=1e-20)
        assert mass_after == pytest.approx(5.0e-6 - deficit)
        # Total (qvv + liquid mass) conserved exactly.
        assert (qvv_after + mass_after) == pytest.approx(-deficit + 5.0e-6)

    def test_concentration_never_touched(self):
        """THE property this test class exists to lock in (code review):
        af_vap's real Fortran algorithm (`cal_mass_budget_vapor`) has no
        `dcondt` anywhere in its body -- `rcon_q` must be bit-for-bit
        unchanged, regardless of how large the vapor deficit is."""
        rmt = np.array([[5.0e-6], [3.0e-6]])
        rcon = np.array([[10.0], [7.5]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        thermo_state = _thermo_state(qvv=np.array([-2.0e-6]))
        config = AmpsConfig.cloudlab()

        liquid_out, _thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        np.testing.assert_array_equal(
            liquid_out.values[lp.rcon_q.py_idx], liquid.values[lp.rcon_q.py_idx]
        )

    def test_rmat_rmas_never_touched(self):
        rmt = np.array([[5.0e-6]])
        rcon = np.array([[10.0]])
        rmat = np.array([[1.0e-8]])
        rmas = np.array([[5.0e-9]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon, rmat=rmat, rmas=rmas)
        thermo_state = _thermo_state(qvv=np.array([-2.0e-6]))
        config = AmpsConfig.cloudlab()

        liquid_out, _thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        assert liquid_out.values[lp.rmat_q.py_idx, 0, 0, 0] == pytest.approx(1.0e-8)
        assert liquid_out.values[lp.rmas_q.py_idx, 0, 0, 0] == pytest.approx(5.0e-9)

    def test_nonnegative_qvv_is_exact_noop(self):
        rmt = np.array([[5.0e-6], [3.0e-6]])
        rcon = np.array([[10.0], [7.5]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        thermo_state = _thermo_state(qvv=np.array([1.0e-2]))
        config = AmpsConfig.cloudlab()

        liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        np.testing.assert_array_equal(liquid_out.values, liquid.values)
        np.testing.assert_array_equal(thermo_out.values, thermo_state.values)

    def test_give_back_distributed_proportionally_to_mass_share(self):
        """G5's own `modc_v` applies the SAME scalar to every bin's
        condensational gain -- this port's closest state-level analogue is
        a proportional-by-mass-share give-back (module docstring): a bin
        with 3x the mass of another gives back 3x as much water."""
        rmt = np.array([[6.0e-6], [2.0e-6]])
        rcon = np.array([[10.0], [10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        deficit = 4.0e-7
        thermo_state = _thermo_state(qvv=np.array([-deficit]))
        config = AmpsConfig.cloudlab()

        liquid_out, _thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        mass_after = liquid_out.values[lp.rmt_q.py_idx, :, 0, 0]
        taken = rmt[:, 0] - mass_after
        # bin 0 has 3x bin 1's mass -> should give back 3x as much.
        assert taken[0] == pytest.approx(3.0 * taken[1])
        assert taken.sum() == pytest.approx(deficit)

    def test_insufficient_liquid_mass_leaves_honest_residual(self):
        """When available liquid mass cannot cover the full deficit, this
        function gives back everything it has and stops (module docstring:
        a genuine physical limit, not a non-convergence bug) -- no
        exception, no infinite loop, and no negative liquid mass."""
        rmt = np.array([[1.0e-7]])
        rcon = np.array([[10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        deficit = 5.0e-6  # far more than the 1e-7 of liquid mass available
        thermo_state = _thermo_state(qvv=np.array([-deficit]))
        config = AmpsConfig.cloudlab()

        liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
        assert liquid_out.values[lp.rmt_q.py_idx, 0, 0, 0] == pytest.approx(0.0, abs=1e-20)
        assert liquid_out.values[lp.rmt_q.py_idx, 0, 0, 0] >= 0.0
        # Residual deficit honestly remains negative (not silently zeroed).
        assert thermo_out.values[qvv_idx, 0, 0, 0] == pytest.approx(-deficit + 1.0e-7)

    def test_no_liquid_mass_at_all_is_a_safe_noop(self):
        """Zero liquid mass everywhere: nothing to give back -- no
        exception, qvv passes through unchanged."""
        rmt = np.array([[0.0]])
        rcon = np.array([[0.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        thermo_state = _thermo_state(qvv=np.array([-1.0e-6]))
        config = AmpsConfig.cloudlab()

        liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
        np.testing.assert_array_equal(liquid_out.values, liquid.values)
        assert thermo_out.values[qvv_idx, 0, 0, 0] == pytest.approx(-1.0e-6)

    def test_does_not_mutate_inputs(self):
        rmt = np.array([[5.0e-6]])
        rcon = np.array([[10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        thermo_state = _thermo_state(qvv=np.array([-1.0e-6]))
        config = AmpsConfig.cloudlab()

        liquid_values_before = liquid.values.copy()
        thermo_values_before = thermo_state.values.copy()

        repair.repair_vapor(liquid, thermo_state, config)

        np.testing.assert_array_equal(liquid.values, liquid_values_before)
        np.testing.assert_array_equal(thermo_state.values, thermo_values_before)

    def test_multipoint_vectorized_independent(self):
        rmt = np.array([[4.0e-6, 6.0e-6]])
        rcon = np.array([[10.0, 10.0]])
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        thermo_state = _thermo_state(qvv=np.array([-1.0e-6, 5.0e-3]))
        config = AmpsConfig.cloudlab()

        liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
        # Point 0: deficit repaired.
        assert thermo_out.values[qvv_idx, 0, 0, 0] == pytest.approx(0.0, abs=1e-20)
        assert liquid_out.values[lp.rmt_q.py_idx, 0, 0, 0] == pytest.approx(4.0e-6 - 1.0e-6)
        # Point 1: already non-negative, untouched.
        assert thermo_out.values[qvv_idx, 0, 0, 1] == pytest.approx(5.0e-3)
        assert liquid_out.values[lp.rmt_q.py_idx, 0, 0, 1] == pytest.approx(6.0e-6)

    def test_finite_result_from_random_state(self):
        rng = np.random.default_rng(4)
        rmt = rng.uniform(0.0, 1.0e-4, size=(20, 5))
        rcon = rng.uniform(0.0, 50.0, size=(20, 5))
        liquid = _liquid_state(rmt=rmt, rcon=rcon)
        qvv = rng.uniform(-1.0e-5, 1.0e-2, size=5)
        thermo_state = _thermo_state(qvv=qvv)
        config = AmpsConfig.cloudlab()

        liquid_out, thermo_out = repair.repair_vapor(liquid, thermo_state, config)

        lp = LiquidPPV
        assert np.all(np.isfinite(liquid_out.values))
        assert np.all(np.isfinite(thermo_out.values))
        assert np.all(liquid_out.values[lp.rmt_q.py_idx] >= 0.0)

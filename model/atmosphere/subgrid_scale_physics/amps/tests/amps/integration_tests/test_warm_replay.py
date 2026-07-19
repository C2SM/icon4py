# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for `driver/box.py`'s warm path (M2a Task 7): `run_box`'s
end-to-end warm-phase driver loop (`case_from_micro_record` ->
`implementations.warm_loop.run_warm_micro_tendency` -> `BoxResult`), per
this task's own dispatch and
docs/superpowers/facts/m2/micro-tendency-orchestration.md ("G1").

Two groups, matching the dispatch's own test list:

* `TestRunBoxSyntheticSmoke` -- UNCONDITIONAL: a synthetic, supersaturated,
  no-pre-existing-liquid `BoxCase` (the same physical scenario as
  `test_warm_loop.py`'s own `TestEndToEndSupersaturatedSpinUp`, here driven
  through the full `box.run_box` entry point instead of
  `run_warm_micro_tendency` directly) executes a full warm step and
  produces a finite, non-negative state. No dump data needed -- runs every
  time.
* `test_warm_replay_against_m0_dump` (`pytest.mark.datatest`) -- the
  per-call replay harness: for every (pre, post) `MicroRecord` pair in a
  local spin-up reference dataset, run `box.run_box` on the "pre" state and
  compare the advanced state to the "post" state, per field, at the
  "~1e-8 warm start" rung of the M2a replay tolerance ladder (this task's
  own dispatch wording -- no finer per-field ladder is established
  elsewhere in the docs as of this task), reporting the single worst
  offender PER FIELD (not merely the first mismatch) if any field exceeds
  tolerance. SKIPPED, with a message pointing at
  docs/superpowers/specs/2026-07-16-ref-data-run-instructions.md, when no
  local dump is found at any of the conventional locations `_find_dump_
  source` checks -- see that function's own docstring. Structured so it
  activates automatically (no code change needed) the moment a dump lands
  at one of those locations.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps, packing
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import load_luts
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.driver import box, ref_data
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    IceState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


try:
    # icon4py-wide test-only dependency, not one of the `amps` package's own
    # runtime dependencies (see that package's `pyproject.toml`) -- guarded so
    # this module still imports cleanly in an environment where it is absent;
    # `_amps_test_data_candidates` below treats that the same as "no
    # conventional ICON4PY_TEST_DATA_PATH/amps/ candidate exists".
    from icon4py.model.testing.config import TEST_DATA_PATH
except ImportError:
    TEST_DATA_PATH = None


# ---------------------------------------------------------------------------
# TestRunBoxSyntheticSmoke -- unconditional, no dump needed.
# ---------------------------------------------------------------------------


def _supersaturated_box_case(*, ptotv_pa: float | None = None) -> box.BoxCase:
    """A cloudlab-shaped (40 liquid bins), single-column, supersaturated,
    no-pre-existing-liquid `BoxCase` -- the exact physical scenario
    `test_warm_loop.py`'s `TestEndToEndSupersaturatedSpinUp` uses
    (`T_STD=280K`, `P_STD=p00`, `qv=1.15e-2` -- proven supersaturated
    there; CCN population `(amt,acon,ams)=(3.164e-13,300.0,3.164e-13)`),
    reused here so `run_box`'s own `moistthermo_mask`-derived `thil`/`qtp`
    (all-zero liquid/ice -> `thp=thv`, `qtp=qvv` exactly, see `run_box`'s
    own docstring) reduce to that same test's own manually-supplied
    `thil=t`/`qtp=qv` -- i.e. `run_box` on this case is a faithful
    superset (adds the `moistthermo_mask` bridge + the ice-mass guard) of
    an already-proven-to-work scenario, not a new, unvalidated one.

    `ptotv_pa`: the SI Pa pressure this fixture's own SI-Exner `thv`
    derivation uses (see below); defaults to `AmpsConst.p00 / 10.0` (the
    CGS `P_STD` this scenario's own physical intent is anchored to, exactly
    SI-Pa-converted -- see `core/packing.py`'s `SCALE_PRE00`/`AmpsConst.
    p00` module docstring notes, both `1e5 Pa`/`1e6 dyn/cm^2` being the
    SAME reference pressure). `ThermoState.ptotv` itself is stored CGS
    (`ptotv_pa * 10.0`, `state.py`'s own UNIT CONTRACT note on
    `ThermoProp.ptotv`, canonicalized at the two `ThermoState` producers --
    this fixture constructs a `ThermoState` directly, so it performs that
    SAME SI->CGS conversion itself, mirroring `core/packing.py`'s
    `_pack_thermo`). `TestRunBoxRealisticPressure` below passes a
    genuinely DIFFERENT value (`90000.0`, cloudlab range, `!= p00`'s SI
    equivalent) -- a regression originally added for the M2a Task 7
    SI-Pa/CGS unit-mismatch code review (see that test's own docstring:
    this exact scenario, with `ptotv` fixed at `p00` in every prior test,
    is what masked the bug in the first place); still exercised post-M2a-
    hardening as a plain "non-p00 realistic pressure" regression, since the
    original bug class is now structurally impossible (conversion happens
    ONCE, at the two producers).

    `thv` is DERIVED from `tv`/`ptotv_pa` via the same SI Exner relation
    `core/packing.py`'s `_pack_thermo` uses (`thv = tv*(SCALE_PRE00/
    ptotv_pa)**(SCALE_RDRY/SCALE_CPDRY)`, using the SI `ptotv_pa`, NOT the
    CGS-converted stored field -- `SCALE_PRE00` is an SI constant), NOT
    hardcoded to `t` -- an earlier draft of this fixture hardcoded `thv=t`
    (valid ONLY when `ptotv_pa == SCALE_PRE00` exactly, the Exner ratio's
    `1.0` fixed point), which silently produced an internally-INCONSISTENT
    state for any other `ptotv_pa` (an ~8K spurious `tv` drift over one
    warm-loop step, caught while calibrating `TestRunBoxRealisticPressure`'s
    own tolerance below -- a fixture bug, not a `run_box` bug; fixed here
    by deriving `thv` properly instead of loosening that test's tolerance
    to paper over it).
    """
    liq_nbins = 40
    p_cgs = float(AmpsConst.p00)
    ptotv_pa = ptotv_pa if ptotv_pa is not None else p_cgs / 10.0
    t = 280.0
    qv = 1.15e-2  # supersaturated at (p_cgs, t)
    thv = t * (packing.SCALE_PRE00 / ptotv_pa) ** (packing.SCALE_RDRY / packing.SCALE_CPDRY)

    thermo_values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: ptotv_pa * 10.0,  # SI Pa -> CGS dyn/cm^2, ThermoState's own UNIT CONTRACT
        ThermoProp.tv: t,
        ThermoProp.thv: thv,
        ThermoProp.piv: t / thv * packing.SCALE_CPDRY,
        ThermoProp.pbv: 0.0,
        # Pre-M2a-hardening this literal (1.2e-3) was stored directly in the
        # nominally-SI moist_denv field, then divided down again by the
        # (now-removed) consumer `* 1.0e-3` conversion -- an inert (no
        # pre-existing liquid bins are active at t=0 here, so `den` does
        # not gate any assertion this class makes) but real actual-physics
        # value of `1.2e-3 * 1.0e-3 = 1.2e-6` g/cm^3. Preserved bit-for-bit
        # here (NOT "fixed" to a realistic magnitude) so this refactor
        # stays behavior-preserving -- see `TestRunBoxRealisticDensity`
        # below for the fixture that DOES use a realistic CGS magnitude.
        ThermoProp.moist_denv: 1.2e-3 * 1.0e-3,
        ThermoProp.qvv: qv,
        ThermoProp.thetav: thv * (1.0 + 0.61 * qv),
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        thermo_values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]

    liquid = LiquidState(
        values=np.zeros((len(LiquidState.PROPS), liq_nbins, 1, 1), dtype=np.float64)
    )
    # No ice group is exercised by the warm path; a small all-zero IceState
    # is enough for moistthermo_mask's own bin loop (run_box's ice-mass
    # guard, box.py's own docstring) -- its bin count is not physically
    # meaningful here, unlike liquid's (which must be a real
    # bin_grid.LIQUID_NBINS count for the real activation/vapor-deposition
    # physics `run_warm_micro_tendency` calls).
    ice = IceState(values=np.zeros((len(IceState.PROPS), 2, 1, 1), dtype=np.float64))

    aerosol_values = np.zeros((len(AerosolState.PROPS), 1, 1, 1), dtype=np.float64)
    ap = index_maps.AerosolPPV
    aerosol_values[ap.amt_q.py_idx, 0, 0, 0] = 3.164e-13
    aerosol_values[ap.acon_q.py_idx, 0, 0, 0] = 300.0
    aerosol_values[ap.ams_q.py_idx, 0, 0, 0] = 3.164e-13
    aerosol = AerosolState(values=aerosol_values)

    config = AmpsConfig.cloudlab()
    assert config.num_h_bins[0] == liq_nbins

    return box.BoxCase(
        thermo=ThermoState(values=thermo_values),
        liquid=liquid,
        ice=ice,
        aerosol=aerosol,
        config=config,
        dt=1.0,
        n_steps=1,
    )


class TestRunBoxSyntheticSmoke:
    def test_runs_full_warm_step_and_produces_finite_nonnegative_state(self) -> None:
        case = _supersaturated_box_case()

        result = box.run_box(case)

        assert np.all(np.isfinite(result.final_liquid.values))
        assert np.all(np.isfinite(result.final_thermo.values))
        assert np.all(np.isfinite(result.final_aerosol.values))

        lp = index_maps.LiquidPPV
        assert np.all(result.final_liquid.values[lp.rmt_q.py_idx] >= 0.0)
        assert np.all(result.final_liquid.values[lp.rcon_q.py_idx] >= 0.0)

    def test_activation_actually_occurred(self) -> None:
        """Sanity: the supersaturated setup is not vacuous -- droplets
        nucleated from CCN (matching the non-negativity assertions above
        not being trivially true of an all-zero state)."""
        case = _supersaturated_box_case()

        result = box.run_box(case)

        lp = index_maps.LiquidPPV
        assert result.final_liquid.values[lp.rmt_q.py_idx].sum() > 0.0

    def test_ice_carried_through_unchanged(self) -> None:
        """The warm path runs no ice process -- `final_ice` must be
        `case`'s own (all-zero) ice state, bit-identical (`box.run_box`'s
        own documented "ice after == ice before" convention)."""
        case = _supersaturated_box_case()

        result = box.run_box(case)

        np.testing.assert_array_equal(result.final_ice.values, case.ice.values)


# ---------------------------------------------------------------------------
# TestRunBoxRealisticPressure -- unconditional regression test, originally
# added for the M2a Task 7 code review: a SI-Pa/CGS unit mismatch
# (`ThermoState.ptotv` was, pre-M2a-hardening, stored SI Pa while
# `core.thermo.diag_t` and every other AMPS-internal CGS-physics formula
# expects CGS dyn/cm^2) was latent since M2a Task 1 (`implementations.
# warm_loop._refresh_state`'s own `diag_t` call passed `ptotv` straight
# through, unconverted) and independently duplicated in `core/liquid_diag.
# py`, `core/vapor_deposition.py`, `core/activation.py` -- ALL of them
# masked by every prior test fixture (including this file's own
# `_supersaturated_box_case`, before that fix) setting `ptotv` numerically
# equal to `AmpsConst.p00` itself, giving a spuriously-exact ratio of 1
# wherever both sides of a mismatched conversion cancelled by construction.
# `run_box` was the first end-to-end path exercised at a REALISTIC pressure
# (`!= p00`'s SI equivalent), which is what actually exposed the bug: with
# it present, `til = thil*(p/p00)**Racp` used a ratio of ~0.1 instead of 1,
# collapsing `tv` by roughly half (e.g. ~270K -> ~140K) for any ptotv in
# the ordinary atmospheric range.
#
# Post-M2a-hardening, `ThermoState.ptotv` is CGS-canonical (`state.py`'s
# own UNIT CONTRACT note, converted ONCE at the two producers) -- the
# original SI-Pa/CGS mismatch bug class is now structurally impossible (no
# per-consumer conversion left to omit). This test remains as a plain
# "non-p00 realistic pressure must not collapse tv" regression, still
# pinning the SAME physical outcome.
# ---------------------------------------------------------------------------


class TestRunBoxRealisticPressure:
    # Cloudlab-range surface pressure, SI Pa, deliberately NOT p00's SI
    # equivalent (1.0e5) -- see class docstring: fixing ptotv AT p00 is
    # exactly what masked the original bug in every prior test.
    REALISTIC_PTOTV_PA = 90_000.0

    def test_tv_stays_physical_no_collapse(self) -> None:
        """A near-adiabatic single step (qr=qi=0 initially, so `_refresh_
        state`'s `til = thil*(p_cgs/p00)**Racp` reconstructs the input `tv`
        EXACTLY when units are handled consistently -- `thil` itself comes
        from `moistthermo_mask`'s `thp = thv/(1+0)` with no liquid/ice
        present, i.e. `thil == thv`, and `thv`/`ptotv_pa` are related by
        the SAME SI Exner relation `core/packing.py`'s `_pack_thermo` uses,
        so the CGS round-trip is a mathematical identity, not a
        coincidence): final `tv` must stay within a few K of the input
        `tv=280.0`, not collapse toward `tv*(0.1)**Racp ~= 145K` (Racp
        ~= 0.286, the ~50% collapse the code review's own bug report
        describes)."""
        t_initial = 280.0
        case = _supersaturated_box_case(ptotv_pa=self.REALISTIC_PTOTV_PA)
        assert case.thermo.values[
            list(ThermoState.PROPS).index(ThermoProp.ptotv), 0, 0, 0
        ] == pytest.approx(self.REALISTIC_PTOTV_PA * 10.0)  # stored CGS, state.py's UNIT CONTRACT

        result = box.run_box(case)

        tv_final = get_thermo_prop(result.final_thermo, ThermoProp.tv)[0]
        assert np.isfinite(tv_final)
        # Generous margin (a few K) for the latent heat a real CCN-
        # activation event releases within one dt=1s step -- NOT a
        # tolerance loose enough to hide a ~50% (~135K) collapse.
        assert tv_final == pytest.approx(t_initial, abs=5.0)


# ---------------------------------------------------------------------------
# TestRunBoxRealisticDensity -- unconditional regression test, originally
# added for the M2a Task 7 code review (second pass): a SECOND, independent
# SI/CGS unit mismatch -- `ThermoState.moist_denv` was, pre-M2a-hardening,
# stored SI kg/m^3, but `core.liquid_diag.diag_pq_liquid`'s
# `_terminal_velocity` compares it (as `den_a`, dry-air density) against
# `AmpsConst.den_w=1.0` (CGS g/cm^3) UNCONVERTED. Reviewer's own repro:
# `den_a=1.2` (realistic SI magnitude) flips `den_w - den_a` negative ->
# `np.log()` of a negative number -> NaN, for any liquid bin in the
# "mid"/"large" cal_terminal_vel_vec regimes (radius > ~10um -- exactly
# where real rain lives). Masked identically to the ptotv bug: every prior
# fixture hardcoded `moist_denv` at CGS magnitude (~1.2e-3), so a missing
# conversion was invisible. See test_liquid_diag.py::TestRealisticDensity
# for the isolated diag_pq_liquid-level repro (that one DOES assert
# `np.isfinite` directly on `terminal_velocity` and failed pre-fix on
# exactly that assertion).
#
# IMPORTANT, established by tracing the actual failure empirically (not
# assumed): the NaN does NOT survive to `run_box`'s own output as a NaN --
# `_ventilation`'s `_ventilation_piecewise` uses `np.select` with NO
# `default=` override, so `x_v=nan` (from `sqrt(nan)`, itself from `nre=
# np.maximum(nan, 0.0)` staying NaN) matches none of `np.select`'s three
# comparison conditions (all NaN comparisons are False) and silently falls
# through to `np.select`'s own implicit default, `0.0` -- NOT NaN. That
# 0.0 then flows into `coef1`/`coef2` (`_vapdep_coef`), silently
# DISABLING vapor-deposition growth for that bin (`d_mean_mass = (coef1*
# s_v_n + coef2)*dt_vp = 0`) rather than corrupting it with NaN. So the
# regression signal at THIS (run_box) level is not `isfinite` (which
# passes regardless, confirmed by first writing this test against the
# unfixed code and finding it a false-negative) -- it is that the
# pre-existing rain bin's own mass must actually CHANGE across the run
# (real vapor-deposition physics ran), not stay frozen at its input value
# (physics silently no-op'd by the laundered-to-zero coefficient).
#
# Post-M2a-hardening, `ThermoState.moist_denv` is CGS-canonical (`state.py`'s
# own UNIT CONTRACT note, converted ONCE at the two producers) -- the
# original SI/CGS mismatch bug class is now structurally impossible. This
# test remains as a plain "realistic density + rain bin -> real vapor-
# deposition physics, not silently disabled" regression, still pinning the
# SAME physical outcome; `_rain_bin_box_case`'s own `moist_denv_si`
# parameter keeps its SI-input semantics (a realistic kg/m^3 magnitude) and
# converts to CGS internally when building the (now CGS-canonical)
# `ThermoState`, mirroring `core/packing.py`'s own producer-side conversion.
# ---------------------------------------------------------------------------

RAIN_BIN = 35  # a large-radius bin in cloudlab's 40-bin liquid grid


def _rain_bin_box_case(*, moist_denv_si: float) -> box.BoxCase:
    """A cloudlab-shaped `BoxCase` with a REALISTIC `moist_denv` (SI
    kg/m^3, this function's own parameter, converted to CGS below when
    building the `ThermoState`) AND a pre-existing rain-size liquid bin
    (200um radius -- `test_liquid_diag.py`'s own `TestRealisticDensity.
    RAIN_RADIUS_CM`, the `cal_terminal_vel_vec` "mid" regime),
    near-saturated (not pushed hard into activation -- this scenario is
    about the pre-existing rain bin's own vapor-deposition/ventilation
    path, not CCN activation)."""
    liq_nbins = 40
    p_pa = 90_000.0  # realistic SI Pa, cloudlab range (also != p00's SI equiv)
    t = 280.0
    qv = 1.0e-2  # near-saturated
    thv = t * (packing.SCALE_PRE00 / p_pa) ** (packing.SCALE_RDRY / packing.SCALE_CPDRY)

    thermo_values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: p_pa * 10.0,  # SI Pa -> CGS dyn/cm^2, ThermoState's own UNIT CONTRACT
        ThermoProp.tv: t,
        ThermoProp.thv: thv,
        ThermoProp.piv: t / thv * packing.SCALE_CPDRY,
        ThermoProp.pbv: 0.0,
        ThermoProp.moist_denv: moist_denv_si * 1.0e-3,  # SI kg/m^3 -> CGS g/cm^3
        ThermoProp.qvv: qv,
        ThermoProp.thetav: thv * (1.0 + 0.61 * qv),
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        thermo_values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]

    lp = index_maps.LiquidPPV
    liquid_values = np.zeros((len(LiquidState.PROPS), liq_nbins, 1, 1), dtype=np.float64)
    radius_cm = 200.0e-4  # 200um -- "mid" cal_terminal_vel_vec regime
    mean_mass = (math.pi / 6.0) * (2.0 * radius_cm) ** 3 * 1.0  # den_w=1.0 g/cm^3, pure water
    con = 1.0
    liquid_values[lp.rcon_q.py_idx, RAIN_BIN, 0, 0] = con
    liquid_values[lp.rmt_q.py_idx, RAIN_BIN, 0, 0] = mean_mass * con
    liquid = LiquidState(values=liquid_values)

    ice = IceState(values=np.zeros((len(IceState.PROPS), 2, 1, 1), dtype=np.float64))

    aerosol_values = np.zeros((len(AerosolState.PROPS), 1, 1, 1), dtype=np.float64)
    ap = index_maps.AerosolPPV
    aerosol_values[ap.amt_q.py_idx, 0, 0, 0] = 3.164e-13
    aerosol_values[ap.acon_q.py_idx, 0, 0, 0] = 300.0
    aerosol_values[ap.ams_q.py_idx, 0, 0, 0] = 3.164e-13
    aerosol = AerosolState(values=aerosol_values)

    config = AmpsConfig.cloudlab()
    assert config.num_h_bins[0] == liq_nbins

    return box.BoxCase(
        thermo=ThermoState(values=thermo_values),
        liquid=liquid,
        ice=ice,
        aerosol=aerosol,
        config=config,
        dt=1.0,
        n_steps=1,
    )


class TestRunBoxRealisticDensity:
    def test_finite_nonnegative_with_realistic_density_and_rain_bin(self) -> None:
        """`moist_denv=1.2` (SI kg/m^3, realistic magnitude, NOT `1.2e-3`
        the CGS magnitude every prior fixture used) combined with a
        pre-existing 200um-radius (rain-size) liquid bin must produce a
        finite, non-negative `BoxResult` -- not NaN."""
        case = _rain_bin_box_case(moist_denv_si=1.2)

        result = box.run_box(case)

        assert np.all(np.isfinite(result.final_liquid.values)), result.final_liquid.values
        assert np.all(np.isfinite(result.final_thermo.values)), result.final_thermo.values
        assert np.all(np.isfinite(result.final_aerosol.values)), result.final_aerosol.values
        lp = index_maps.LiquidPPV
        assert np.all(result.final_liquid.values[lp.rmt_q.py_idx] >= 0.0)
        assert np.all(result.final_liquid.values[lp.rcon_q.py_idx] >= 0.0)

    def test_rain_bin_vapor_deposition_actually_ran(self) -> None:
        """THE discriminating assertion (see class docstring above: the
        bug does NOT surface as a NaN in `run_box`'s own output -- it
        surfaces as `np.select`'s NaN-comparisons-are-False fallback
        silently zeroing `coef1`/`coef2`, which silently DISABLES vapor-
        deposition growth for the rain bin). Verified empirically against
        the unfixed code (temporarily reverted `core/liquid_diag.py` +
        `core/activation.py` via `git stash`, this exact scenario): the
        rain bin's `rmt` came back bit-IDENTICAL to its input (`3.35103e-05
        == 3.35103e-05`, physics silently no-op'd); with the fix, it comes
        back measurably different (`3.35527e-05`, real Chen-Lamb growth)."""
        case = _rain_bin_box_case(moist_denv_si=1.2)
        lp = index_maps.LiquidPPV
        rmt_before = float(case.liquid.values[lp.rmt_q.py_idx, RAIN_BIN, 0, 0])
        assert rmt_before > 0.0  # sanity: the rain bin fixture is non-vacuous

        result = box.run_box(case)

        rmt_after = float(result.final_liquid.values[lp.rmt_q.py_idx, RAIN_BIN, 0, 0])
        assert np.isfinite(rmt_after)
        assert rmt_after != pytest.approx(rmt_before, rel=1.0e-6), (
            f"rain bin mass unchanged ({rmt_before!r} -> {rmt_after!r}) -- vapor-deposition "
            "physics silently no-op'd (moist_denv SI/CGS bug regressed)"
        )

    @pytest.mark.parametrize("moist_denv_si", [1.0, 1.1, 1.2])
    def test_finite_across_realistic_density_range(self, moist_denv_si) -> None:
        case = _rain_bin_box_case(moist_denv_si=moist_denv_si)

        result = box.run_box(case)

        assert np.all(np.isfinite(result.final_liquid.values))
        assert np.all(np.isfinite(result.final_thermo.values))


# ---------------------------------------------------------------------------
# Per-call replay against a real scale_amps M0 dump (marker-gated).
# ---------------------------------------------------------------------------

_AMPS_DUMP_DIR_ENV = "AMPS_DUMP_DIR"

# "~1e-8 warm start" rung of the M2a replay tolerance ladder (this task's own
# dispatch wording) -- one relative tolerance, floored by a small absolute
# tolerance so near-zero bins (most liquid bins carry no mass at any given
# instant) don't spuriously fail on relative terms alone.
WARM_REPLAY_RTOL = 1.0e-8
WARM_REPLAY_ATOL = 1.0e-12


def _amps_test_data_candidates() -> list[Path]:
    """Conventional `$ICON4PY_TEST_DATA_PATH/amps/` locations checked by
    `_find_dump_source` -- both the raw-dump-directory and converted-`.npz`
    forms `driver.ref_data.load_reference` accepts (see that function's own
    docstring). Returns an empty list if `icon4py.model.testing` was not
    importable at module load time (see the guarded import above), rather
    than raising -- `_find_dump_source` treats that the same as "no
    candidate exists"."""
    if TEST_DATA_PATH is None:
        return []
    return [
        TEST_DATA_PATH / "amps" / "amps_dump",
        TEST_DATA_PATH / "amps" / "amps_ref_spinup.npz",
    ]


def _find_dump_source() -> Path | None:
    """Locate a local AMPS M0 reference-data source -- a directory of raw
    `amps_dump_r*_t*.bin` files, or one converted `.npz` archive -- per
    docs/superpowers/specs/2026-07-16-ref-data-run-instructions.md ("§4.
    Collect results": `amps_dump_reader.py amps_dump/ -o
    amps_ref_<runname>.npz`). Either form is accepted directly by
    `ref_data.load_reference`.

    Checked, in order (first hit wins):

    1. `$AMPS_DUMP_DIR`, if set: an explicit override, either a raw dump
       directory or a single `.npz` file. A set-but-missing path is
       treated as absent (returns `None`, not a silent fall-through to
       #2/#3 below) -- a typo'd override should surface as "no dump
       found", not silently resolve to some other, unintended dataset.
    2. `$ICON4PY_TEST_DATA_PATH/amps/amps_dump` (a raw dump directory).
    3. `$ICON4PY_TEST_DATA_PATH/amps/amps_ref_spinup.npz` (a converted
       archive; `amps_ref_spinup` is the exact artifact name the spec's
       own collection step produces for the primary "warm spin-up"
       validation target -- the spec's `§3.1`, `run.conf`).

    Returns `None` if nothing is found at any of the above -- the caller
    (`test_warm_replay_against_m0_dump`) skips with `_SKIP_MESSAGE` in
    that case.
    """
    env_path = os.environ.get(_AMPS_DUMP_DIR_ENV)
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.exists() else None
    for candidate in _amps_test_data_candidates():
        if candidate.exists():
            return candidate
    return None


_SKIP_MESSAGE = (
    "No local AMPS M0 reference-data dump found for the warm-replay harness -- checked "
    f"${_AMPS_DUMP_DIR_ENV}, $ICON4PY_TEST_DATA_PATH/amps/amps_dump, and "
    "$ICON4PY_TEST_DATA_PATH/amps/amps_ref_spinup.npz. Produce one per "
    "docs/superpowers/specs/2026-07-16-ref-data-run-instructions.md (scale_amps repo: a "
    "cluster run of scale-rm/test/case/cloudlab/scripts/run.conf with l_amps_dump=.true., "
    "then locally `python3 scripts/amps_dump_reader.py amps_dump/ -o amps_ref_spinup.npz`), "
    f"then either set ${_AMPS_DUMP_DIR_ENV} to the dump directory/.npz path, or place it at "
    "$ICON4PY_TEST_DATA_PATH/amps/, and re-run -- this test activates automatically once "
    "found, no code change needed."
)


@dataclasses.dataclass(frozen=True)
class _FieldMismatch:
    """One field's single worst-offending column point, across every
    replayed (pre, post) pair -- `test_warm_replay_against_m0_dump`
    tracks (at most) one of these PER FIELD, not one per mismatched point
    (a real dump may have thousands of column points; reporting only the
    worst keeps a failure message readable)."""

    pair_key: tuple[int, int, int, int]  # (rank, TIME_AMPS, i, j)
    field: str
    index: tuple[int, ...]
    actual: float
    expected: float
    abs_err: float
    rel_err: float


def _worst_point_mismatch(
    pair_key: tuple[int, int, int, int],
    field: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> _FieldMismatch | None:
    """The worst (largest absolute-error) out-of-tolerance point in
    `actual` vs. `expected` for one field of one (pre, post) pair, or
    `None` if every point is within `atol + rtol*|expected|` (numpy's own
    `allclose` tolerance convention). A shape mismatch is reported as an
    (always-worst, `abs_err=inf`) mismatch rather than raising -- it is
    itself the finding this harness exists to catch."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return _FieldMismatch(
            pair_key=pair_key,
            field=f"{field} (SHAPE MISMATCH: actual={actual.shape} expected={expected.shape})",
            index=(),
            actual=float("nan"),
            expected=float("nan"),
            abs_err=float("inf"),
            rel_err=float("inf"),
        )

    abs_err = np.abs(actual - expected)
    tol = atol + rtol * np.abs(expected)
    bad = abs_err > tol
    if not np.any(bad):
        return None

    flat_idx = int(np.argmax(np.where(bad, abs_err, -np.inf)))
    idx = np.unravel_index(flat_idx, actual.shape)
    rel = abs_err[idx] / max(abs(float(expected[idx])), atol)
    return _FieldMismatch(
        pair_key=pair_key,
        field=field,
        index=idx,
        actual=float(actual[idx]),
        expected=float(expected[idx]),
        abs_err=float(abs_err[idx]),
        rel_err=float(rel),
    )


def _update_worst(worst: dict[str, _FieldMismatch], candidate: _FieldMismatch | None) -> None:
    if candidate is None:
        return
    current = worst.get(candidate.field)
    if current is None or candidate.abs_err > current.abs_err:
        worst[candidate.field] = candidate


@pytest.mark.datatest
def test_warm_replay_against_m0_dump() -> None:
    """See module docstring. Compared fields -- `liquid.values`,
    `aerosol.values`, `thermo.tv`, `thermo.qvv` -- are exactly the ones
    `box.run_box`'s own docstring documents as genuinely advanced by the
    warm loop; `thermo.thv`/`piv`/`moist_denv` are deliberately excluded
    (a documented `implementations.warm_loop` scope gap -- `update_
    airgroup` has no port -- not something to (mis)validate here).

    SCOPE CAVEAT (M2a Task 7 code review): this function's own logic was
    verified end-to-end with a SELF-referential round-trip -- a synthetic
    "post" `MicroRecord` built FROM `run_box`'s own output on a synthetic
    "pre" record, fed back through this exact function, both to confirm
    the pass path (exact match by construction) and the worst-offender
    failure-reporting path (via a deliberately corrupted field). That
    round-trip validates ONLY this harness's OWN bookkeeping -- dump
    loading, pairing, field comparison, worst-offender tracking -- it
    CANNOT catch a systematic error in `run_box`'s physics itself (the
    same error, present on both the "actual" and "expected" side by
    construction, trivially cancels; see the SI-Pa/CGS pressure bug this
    same code review caught, which such a self-referential check would
    NOT have exposed). Genuine physics validation requires this test to
    run against an EXTERNAL scale_amps Fortran dump (see `_find_dump_
    source`'s own docstring) -- something no local checkout has yet."""
    dump_source = _find_dump_source()
    if dump_source is None:
        pytest.skip(_SKIP_MESSAGE)

    dataset = ref_data.load_reference(dump_source)
    luts = load_luts()

    worst_per_field: dict[str, _FieldMismatch] = {}
    n_compared = 0
    n_ice_skipped = 0

    for pre, post in dataset.micro_pairs():
        pair_key = (pre.rank, pre.TIME_AMPS, pre.i, pre.j)
        case = box.case_from_micro_record(pre)
        try:
            result = box.run_box(case, luts=luts)
        except NotImplementedError:
            # Ice/mixed-phase column -- M3 scope (box.run_box's own
            # docstring). The warm spin-up is documented to carry zero ice
            # mass throughout
            # (docs/superpowers/specs/2026-07-16-ref-data-run-instructions.md
            # §3.1: ice routines execute and no-op, never nucleate), so
            # this is not expected to fire against a real spin-up dump;
            # skipped rather than failed so an ice-bearing dump (e.g. the
            # seeding run, out of THIS task's scope) doesn't crash the
            # harness outright.
            n_ice_skipped += 1
            continue

        n_compared += 1
        for field, actual, expected in (
            ("liquid", result.final_liquid.values, post.qrpvm),
            ("aerosol", result.final_aerosol.values, post.qapvm),
            ("thermo.tv", get_thermo_prop(result.final_thermo, ThermoProp.tv), post.tvm),
            ("thermo.qvv", get_thermo_prop(result.final_thermo, ThermoProp.qvv), post.qvvm),
        ):
            _update_worst(
                worst_per_field,
                _worst_point_mismatch(
                    pair_key, field, actual, expected, rtol=WARM_REPLAY_RTOL, atol=WARM_REPLAY_ATOL
                ),
            )

    assert n_compared > 0, (
        f"dump source {dump_source} yielded zero warm (ice-free) micro_pairs() to replay "
        f"({n_ice_skipped} ice-bearing pair(s) skipped) -- nothing was validated"
    )

    if worst_per_field:
        lines = [
            f"  field={m.field} pair(rank,t,i,j)={m.pair_key} idx={m.index}: "
            f"actual={m.actual!r} expected={m.expected!r} "
            f"abs_err={m.abs_err:.3e} rel_err={m.rel_err:.3e}"
            for m in worst_per_field.values()
        ]
        pytest.fail(
            f"warm-loop replay: {len(worst_per_field)} field(s) exceeded "
            f"rtol={WARM_REPLAY_RTOL:.0e}/atol={WARM_REPLAY_ATOL:.0e} across {n_compared} "
            f"pair(s) ({n_ice_skipped} ice-bearing pair(s) skipped). Worst offender per field:\n"
            + "\n".join(lines)
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/activation.py (M2a Task 4, "Part 2"): the outer driver
`activate_and_advance_vapor` (`cal_aptact_var8_kc04dep`), per
docs/superpowers/facts/m2/activation.md ("G2") sections 1a-1n / 3.

Groups, matching the task brief's test list:

* `TestCcnPrecomputeHandKohler` -- "number of activated droplets vs a
  hand-computed Koehler case": `_golden_ccn_bin` is a scalar (`math`, not
  numpy) independent re-derivation of `_ccn_precompute`'s per-quadrature-
  bin Koehler activation math (G2 section 1c + `diag_pardis_ap`,
  `class_Mass_Bin.F90:1743-1789`), calling already-independently-tested
  Part 1 primitives (`kohler_critical_radius`) directly -- matching
  `test_activation_solver.py`'s own precedent ("the reference ... hand-
  codes the G2 formulas themselves, not a copy of the module-under-test's
  numpy code") and `test_liquid_diag.py`'s `_golden_bin`.
* `TestMassConservation` -- vapor+condensate mass conservation across an
  activation step (a pure-nucleation scenario, no pre-existing liquid, so
  the conservation identity is EXACT -- see `core/activation.py`'s own
  `activate_and_advance_vapor` docstring for why this driver's own output
  is conservative in isolation only when there is no pre-existing
  condensational growth to account for; `vapor_deposition`, M2a Task 5,
  owns that separately).
* `TestDropletPlacement` -- activated droplets land in the liquid bin
  whose mass range contains their own grown mean mass (`add_simple_vec`).
* `TestDhfToggle` -- `config.ice_nucleation_dhf` toggles the DHF
  precompute's `NotImplementedError`/no-op behavior.
* `TestDepositionSafeDefault` -- the deposition-nucleation precompute is a
  safe no-op (never raises) when no dust (category-2) aerosol population
  is present, regardless of `config.ice_nucleation_deposition`.
* `TestSkipMask` -- an all-skipped grid box (`icycle_n==1` everywhere)
  returns the input state unchanged.
* `test_activation_replay_against_m0_dump` (`pytest.mark.datatest`) --
  SKIPPED with a pointer: no local scale_amps M0 per-call activation dumps
  exist in this checkout (`driver/ref_data.py` can load
  `amps_dump_r*.bin` if produced by a real scale_amps DEBUG run; none are
  committed here, matching `test_ref_data.py`'s own synthesized-only
  fixtures).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import (
    activation,
    bin_grid,
    index_maps,
    liquid_diag,
    thermo,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
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


P_STD = float(AmpsConst.p00)  # 1e6 dyn/cm^2 ~ 1000 hPa
T_STD = 280.0  # K
DEN_STD = 1.2e-3  # g/cm^3, moist air

# cloudlab's own liquid bin grid: 40 bins, haze split at 20 -- required by
# bin_grid.make_bin_grid (LIQUID_NBINS = (40, 80) only).
LIQ_NBINS = 40
NBIN_H = 20


def _thermo_state(*, p: float, t: float, den: float, qv: float) -> ThermoState:
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: p,
        ThermoProp.tv: t,
        ThermoProp.thv: t,
        ThermoProp.piv: 0.0,
        ThermoProp.pbv: 0.0,
        ThermoProp.moist_denv: den,
        ThermoProp.qvv: qv,
        ThermoProp.thetav: t,
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]
    return ThermoState(values=values)


def _liquid_state_40bin(populated: dict[int, tuple[float, float, float, float]]) -> LiquidState:
    """40-bin, single-column `LiquidState`; `populated`: {bin_index:
    (rmt, rcon, rmat, rmas)}, all other bins zero."""
    lp = index_maps.LiquidPPV
    values = np.zeros((len(LiquidState.PROPS), LIQ_NBINS, 1, 1), dtype=np.float64)
    for b, (rmt, rcon, rmat, rmas) in populated.items():
        values[lp.rmt_q.py_idx, b, 0, 0] = rmt
        values[lp.rcon_q.py_idx, b, 0, 0] = rcon
        values[lp.rmat_q.py_idx, b, 0, 0] = rmat
        values[lp.rmas_q.py_idx, b, 0, 0] = rmas
    return LiquidState(values=values)


def _aerosol_state_single_bin(cat0: tuple[float, float, float], ncat: int = 1) -> AerosolState:
    """Single-column, `nbins=1` `AerosolState` (matching the Fortran's own
    `ga(ica)%MS(1,n)` bulk-bin-per-category convention, see
    `core/activation.py`'s `_ccn_precompute` docstring): `cat0` = (amt,
    acon, ams) for category 0; all other categories (if `ncat>1`) zero."""
    values = np.zeros((len(AerosolState.PROPS), 1, ncat, 1), dtype=np.float64)
    values[0, 0, 0, 0] = cat0[0]
    values[1, 0, 0, 0] = cat0[1]
    values[2, 0, 0, 0] = cat0[2]
    return AerosolState(values=values)


def _diag_for(liquid: LiquidState, thermo_state: ThermoState, config: AmpsConfig, luts_: AmpsLuts):
    return liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts_)


def _base_config(**overrides) -> AmpsConfig:
    base = AmpsConfig.cloudlab()
    assert base.num_h_bins[0] == LIQ_NBINS
    assert base.nbin_h == NBIN_H
    return dataclasses.replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# _ccn_precompute -- hand-computed Koehler case.
# ---------------------------------------------------------------------------


def _golden_ccn_bin(
    i: int,
    *,
    amt1: float,
    acon1: float,
    ap_lnsig: float,
    den_as: float,
    den_ai: float,
    eps: float,
    nu_aps: float,
    phi_aps: float,
    m_aps: float,
    aa_over_t: float,
    liquid_binb1: float,
    snrml,
) -> tuple[float, float, float, float]:
    """Independent scalar (`math`) re-derivation of `_ccn_precompute`'s
    per-quadrature-bin Koehler activation math for ONE bin `i` (G2 section
    1c + `diag_pardis_ap`), calling Part 1's already independently-tested
    `kohler_critical_radius` directly (not `_ccn_precompute`'s own
    computation) -- see module docstring. Uses `_interp_data1d_lut_big`
    (not raw `scipy.stats.norm.sf`) for the survival-function lookup:
    `snrml`'s own grid only spans `x=[0,5]` (F3 SS5.2), so at the extreme
    tail quadrature bins (`|xi|>5`) the LUT's own boundary clamp matters
    -- reusing the already-independently-derived-from-Fortran clamping
    logic here (not re-deriving it a second time) matches this module's
    own precedent for calling already-tested infrastructure directly.

    Returns `(n_act_i, m_act_i, s_c, mean_mass_grown)`.
    """
    coef4pi3 = float(AmpsConst.coef4pi3)
    den_w = float(AmpsConst.den_w)
    m_w = float(AmpsConst.M_w)

    den_ap = den_ai / (1.0 - eps * (1.0 - den_ai / den_as))
    mean_mass_bulk = amt1 / acon1
    log_arg = mean_mass_bulk / (coef4pi3 * den_ap)
    p1 = math.exp((math.log(log_arg) - 4.5 * ap_lnsig**2) / 3.0)
    del p1  # only needed for `dum1` (unused downstream, see module docstring)

    zn = activation.ZN_QUADRATURE
    fact_c = activation.FACT_C_QUADRATURE
    n_act_i = float(fact_c[i]) * acon1

    xi1 = float(zn[i]) - 3.0 * ap_lnsig
    xi2 = float(zn[i + 1]) - 3.0 * ap_lnsig
    yi1 = float(activation._interp_data1d_lut_big(snrml, np.array([abs(xi1)]))[0])
    yi2 = float(activation._interp_data1d_lut_big(snrml, np.array([abs(xi2)]))[0])
    if xi1 >= 0.0:
        fm2 = yi1 - yi2
    elif xi2 > 0.0:
        fm2 = 1.0 - yi1 - yi2
    else:
        fm2 = yi2 - yi1
    fact_m = min(1.0, fm2)
    m_act_i = fact_m * mean_mass_bulk * acon1

    mean_mass_ap = m_act_i / n_act_i
    r_n_micron = 1.0e4 * (mean_mass_ap / den_ap / coef4pi3) ** (1.0 / 3.0)
    sb = nu_aps * eps * m_w * den_ap / (m_aps * den_w) * phi_aps
    beta = 0.5
    r_n_cm = r_n_micron * 1.0e-4

    rd_c_micron = 1.0e4 * float(activation.kohler_critical_radius(aa_over_t, sb, beta, r_n_cm))
    s_c = (
        math.exp(
            aa_over_t * 1.0e4 / rd_c_micron
            - sb * r_n_micron ** (2.0 * (1.0 + beta)) / (rd_c_micron**3 - r_n_micron**3)
        )
        - 1.0
    )

    mean_mass_grown = max(liquid_binb1 * 1.05, mean_mass_ap * 1.05)
    return n_act_i, m_act_i, s_c, mean_mass_grown


class TestCcnPrecomputeHandKohler:
    """`_ccn_precompute`'s output for a known aerosol population, cross-
    checked bin-by-bin against `_golden_ccn_bin`."""

    AMT1 = 3.164e-13  # g/cm^3, total aerosol mass, category 0
    ACON1 = 300.0  # #/cm^3, number concentration
    AP_LNSIG = 0.712949807856125  # cloudlab category-0 value (config.ap_lnsig[0])

    def _setup(self, sw: float):
        config = _base_config()
        aerosol = _aerosol_state_single_bin(
            (self.AMT1, self.ACON1, self.AMT1)
        )  # eps=1 (fully soluble)
        t = np.array([T_STD])
        sig_wa = thermo.sfc_tension(t)
        liquid_binb = bin_grid.make_bin_grid("liquid", LIQ_NBINS, nbin_h=NBIN_H).binb
        icycle_active = np.array([True])
        sw_arr = np.array([sw])
        ccn = activation._ccn_precompute(
            aerosol,
            (0,),
            config=config,
            luts=load_luts(),
            icycle_active=icycle_active,
            sw=sw_arr,
            t=t,
            sig_wa=sig_wa,
            liquid_binb1=float(liquid_binb[0]),
        )
        coef_a = 2.0 / (float(AmpsConst.den_w) * float(AmpsConst.R_v))
        aa_over_t = coef_a * float(sig_wa[0]) / T_STD
        return ccn, aa_over_t, float(liquid_binb[0]), config

    def test_matches_hand_derivation_all_bins(self) -> None:
        ccn, aa_over_t, binb1, config = self._setup(sw=0.02)
        snrml = load_luts().snrml
        for i in range(activation.N_BIN_A):
            n_act_i, m_act_i, s_c, mean_mass_grown = _golden_ccn_bin(
                i,
                amt1=self.AMT1,
                acon1=self.ACON1,
                ap_lnsig=self.AP_LNSIG,
                den_as=float(config.den_aps[0]),
                den_ai=float(config.den_api[0]),
                eps=1.0,
                nu_aps=float(config.nu_aps[0]),
                phi_aps=float(config.phi_aps[0]),
                m_aps=float(config.M_aps[0]),
                aa_over_t=aa_over_t,
                liquid_binb1=binb1,
                snrml=snrml,
            )
            np.testing.assert_allclose(ccn.s_c[i, 0, 0], s_c, rtol=1.0e-6)
            # n_act/m_act are only populated (nonzero) when the bin actually
            # activates (noccn0) -- compare directly against the RAW
            # quadrature values regardless of the gate, computed the same way.
            if ccn.noccn0[i, 0, 0]:
                np.testing.assert_allclose(ccn.n_act[i, 0, 0], n_act_i, rtol=1.0e-6)
                np.testing.assert_allclose(ccn.m_act[i, 0, 0], m_act_i, rtol=1.0e-6)
                np.testing.assert_allclose(
                    ccn.mean_mass_grown[i, 0, 0], mean_mass_grown, rtol=1.0e-6
                )
            else:
                assert s_c > 0.02 or n_act_i < NLMT_TEST or m_act_i < NLMT_TEST

    def test_activation_gate_toggles_with_supersaturation(self) -> None:
        """One quadrature bin's own hand-computed `s_c`: below it the bin
        must NOT activate, (well) above it it MUST."""
        ccn_low, _, _, _ = self._setup(sw=1.0e-6)
        ccn_high, _, _, _ = self._setup(sw=0.02)
        # At vanishingly small sw, no bin should activate (s_c > sw for any
        # physically reasonable CCN population).
        assert not ccn_low.noccn0.any()
        # At sw=0.02 (=SW_ALLOW, the ceiling), at least the largest-particle
        # quadrature bins (smallest s_c) should activate.
        assert ccn_high.noccn0.any()
        # Every activated bin's own s_c must be <= sw, by construction of the gate.
        activated = ccn_high.noccn0[:, 0, 0]
        assert np.all(ccn_high.s_c[:, 0, 0][activated] <= 0.02 + 1.0e-12)


NLMT_TEST = 1.0e-30


# ---------------------------------------------------------------------------
# Mass conservation (vapor + condensate) across an activation step.
# ---------------------------------------------------------------------------


class TestMassConservation:
    """Pure-nucleation scenario (no pre-existing liquid): `qv + (rmt-rmat)`
    total-water mixing ratio is EXACTLY conserved -- see
    `activate_and_advance_vapor`'s own docstring for why this identity
    only holds exactly when there is no pre-existing condensate to grow
    (`used_Mr_vap`, owned by the separate `vapor_deposition` process,
    M2a Task 5, not this driver)."""

    def test_water_mass_conserved(self) -> None:
        config = _base_config()
        luts_ = load_luts()

        qv = 1.15e-2  # supersaturated relative to T_STD/P_STD, see below
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=qv)
        liquid = _liquid_state_40bin({})  # no pre-existing droplets
        aerosol = _aerosol_state_single_bin((3.164e-13, 300.0, 3.164e-13))

        diag = _diag_for(liquid, thermo_state, config, luts_)

        # Sanity: this scenario is indeed supersaturated (sw>0), else no
        # activation would occur and the test would be vacuous.
        estbar, esitbar = thermo.make_esat_tables()
        sw0 = activation._liquid_supersaturation(
            np.array([P_STD]), np.array([qv]), np.array([T_STD]), estbar, esitbar
        )
        assert sw0[0] > 0.0

        liquid_after, _aerosol_after, thermo_after = activation.activate_and_advance_vapor(
            liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
        )

        qv_after = float(
            thermo_after.values[list(ThermoState.PROPS).index(ThermoProp.qvv), 0, 0, 0]
        )
        lp = index_maps.LiquidPPV
        water_after = (
            float(
                (liquid_after.values[lp.rmt_q.py_idx] - liquid_after.values[lp.rmat_q.py_idx]).sum()
            )
            / DEN_STD
        )

        # Some activation must actually have happened for this to be a
        # meaningful test (not a vacuous 0==0).
        assert water_after > 0.0

        np.testing.assert_allclose(qv_after + water_after, qv, rtol=1.0e-9)

    def test_no_activation_is_a_no_op(self) -> None:
        """Sub-saturated box (sw<=0): the grid-box skip mask (G2 section
        1a) should leave the state completely unchanged."""
        config = _base_config()
        luts_ = load_luts()
        qv = 1.0e-4  # far sub-saturated at T_STD/P_STD
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=qv)
        liquid = _liquid_state_40bin({})
        aerosol = _aerosol_state_single_bin((3.164e-13, 300.0, 3.164e-13))
        diag = _diag_for(liquid, thermo_state, config, luts_)

        liquid_after, aerosol_after, thermo_after = activation.activate_and_advance_vapor(
            liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
        )
        np.testing.assert_array_equal(liquid_after.values, liquid.values)
        np.testing.assert_array_equal(aerosol_after.values, aerosol.values)
        np.testing.assert_array_equal(thermo_after.values, thermo_state.values)


# ---------------------------------------------------------------------------
# Droplet placement into the correct liquid bin (add_simple_vec).
# ---------------------------------------------------------------------------


class TestDropletPlacement:
    def test_activated_mass_lands_in_the_bin_containing_its_mean_mass(self) -> None:
        config = _base_config()
        luts_ = load_luts()
        qv = 1.15e-2
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=qv)
        liquid = _liquid_state_40bin({})
        aerosol = _aerosol_state_single_bin((3.164e-13, 300.0, 3.164e-13))
        diag = _diag_for(liquid, thermo_state, config, luts_)

        liquid_after, _, _ = activation.activate_and_advance_vapor(
            liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
        )

        lp = index_maps.LiquidPPV
        rmt = liquid_after.values[lp.rmt_q.py_idx, :, 0, 0]
        rcon = liquid_after.values[lp.rcon_q.py_idx, :, 0, 0]
        populated = np.nonzero(rcon > 0.0)[0]
        assert populated.size > 0

        binb = bin_grid.make_bin_grid("liquid", LIQ_NBINS, nbin_h=NBIN_H).binb
        for j in populated:
            mean_mass = rmt[j] / rcon[j]
            # add_simple_vec's own placement invariant (mod_amps_core.F90:
            # 15507-15511): binb[j] < mean_mass <= binb[j+1], OR mean_mass
            # is clamped to just outside that (the Np clamp allows the
            # PLACED mean mass -- using the CLAMPED Np, not the original --
            # to sit within [0.99*binb[j], 1.01*binb[j+1]]).
            assert 0.99 * binb[j] <= mean_mass <= 1.01 * binb[j + 1]


# ---------------------------------------------------------------------------
# DHF toggle.
# ---------------------------------------------------------------------------


class TestDhfToggle:
    def _scenario(self, config: AmpsConfig):
        luts_ = load_luts()
        qv = 1.15e-2
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=qv)
        liquid = _liquid_state_40bin({})
        aerosol = _aerosol_state_single_bin((3.164e-13, 300.0, 3.164e-13))
        diag = _diag_for(liquid, thermo_state, config, luts_)
        return liquid, aerosol, thermo_state, luts_, diag

    def test_dhf_off_runs_cleanly(self) -> None:
        config = _base_config(ice_nucleation_dhf=0)
        liquid, aerosol, thermo_state, luts_, diag = self._scenario(config)
        # Must not raise.
        activation.activate_and_advance_vapor(
            liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
        )

    def test_dhf_on_raises_without_placeholder(self) -> None:
        config = _base_config(ice_nucleation_dhf=1)
        liquid, aerosol, thermo_state, luts_, diag = self._scenario(config)
        with pytest.raises(NotImplementedError, match="DHF_IF1"):
            activation.activate_and_advance_vapor(
                liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
            )

    def test_dhf_on_with_placeholder_runs_cleanly(self) -> None:
        config = _base_config(ice_nucleation_dhf=1)
        liquid, aerosol, thermo_state, luts_, diag = self._scenario(config)
        # Must not raise, and (no ice group at all in this scenario) must
        # produce the SAME result as DHF fully off -- the placeholder is a
        # provable no-op.
        liquid_off, aerosol_off, thermo_off = activation.activate_and_advance_vapor(
            liquid,
            aerosol,
            thermo_state,
            _base_config(ice_nucleation_dhf=0),
            dt_vp=1.0,
            luts=luts_,
            diag=diag,
        )
        liquid_on, aerosol_on, thermo_on = activation.activate_and_advance_vapor(
            liquid,
            aerosol,
            thermo_state,
            config,
            dt_vp=1.0,
            luts=luts_,
            diag=diag,
            allow_dhf_placeholder=True,
        )
        np.testing.assert_allclose(liquid_on.values, liquid_off.values)
        np.testing.assert_allclose(aerosol_on.values, aerosol_off.values)
        np.testing.assert_allclose(thermo_on.values, thermo_off.values)


# ---------------------------------------------------------------------------
# Deposition-nucleation: safe no-op with no dust category present.
# ---------------------------------------------------------------------------


class TestDepositionSafeDefault:
    def test_deposition_on_with_no_dust_category_never_raises(self) -> None:
        """cloudlab's own `ice_nucleation_deposition=True` (iflg_dep=1) --
        must NOT raise when the aerosol state has no category-2 (dust)
        population (`ncat=1` here), matching a warm-only wiring with no
        dust group supplied (see `_deposition_precompute`'s docstring)."""
        config = _base_config()  # ice_nucleation_deposition=True (cloudlab default)
        assert config.ice_nucleation_deposition
        luts_ = load_luts()
        qv = 1.15e-2
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=qv)
        liquid = _liquid_state_40bin({})
        aerosol = _aerosol_state_single_bin((3.164e-13, 300.0, 3.164e-13), ncat=1)
        diag = _diag_for(liquid, thermo_state, config, luts_)

        activation.activate_and_advance_vapor(
            liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
        )


# ---------------------------------------------------------------------------
# Grid-box skip mask.
# ---------------------------------------------------------------------------


class TestSkipMask:
    def test_no_water_box_returns_input_unchanged(self) -> None:
        """`mes_rc==0` (no water at all): `icycle_n=1`, the box is
        entirely skipped -- the early-return path (G2 section 1a)."""
        config = _base_config()
        luts_ = load_luts()
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=0.0)
        liquid = _liquid_state_40bin({})
        aerosol = _aerosol_state_single_bin((0.0, 0.0, 0.0))
        diag = _diag_for(liquid, thermo_state, config, luts_)

        liquid_after, aerosol_after, thermo_after = activation.activate_and_advance_vapor(
            liquid, aerosol, thermo_state, config, dt_vp=1.0, luts=luts_, diag=diag
        )
        assert liquid_after is liquid
        assert aerosol_after is aerosol
        assert thermo_after is thermo_state


# ---------------------------------------------------------------------------
# Per-call replay against a real scale_amps M0 dump (marker-gated).
# ---------------------------------------------------------------------------


@pytest.mark.datatest
def test_activation_replay_against_m0_dump() -> None:
    """Would spin up a pre-recorded aerosol+thermo state (scale_amps M0
    per-call DEBUG dump), run `activate_and_advance_vapor`, and compare the
    resulting activated liquid + advanced vapor against the recorded
    post-activation state (rtol ~1e-8). SKIPPED: no local scale_amps M0
    per-call activation dumps exist in this checkout (`driver/ref_data.py`
    can load `amps_dump_r*.bin` if produced by a real scale_amps DEBUG
    run -- see that module's `read_dump_file`/`load_reference` -- none are
    committed here; `test_ref_data.py` itself only exercises synthesized
    in-memory fixtures, never a real dump directory)."""
    pytest.skip(
        "No local scale_amps M0 per-call activation dumps available in this checkout -- "
        "see driver/ref_data.py (read_dump_file/load_reference) for the loader once real "
        "amps_dump_r*.bin files (DEBUG-mode scale_amps run, activation call site) exist."
    )

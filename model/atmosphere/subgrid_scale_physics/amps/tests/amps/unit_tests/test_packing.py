# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/packing.py (M1 Task 7): SI<->CGS pack/unpack between SCALE
and AMPS state, per docs/superpowers/facts/m1/state-packing-si-cgs.md ("F4").

Three groups, matching the task brief:
* TestPackThermo / TestPackLiquid / TestPackIce / TestPackAerosol / TestPackEmoist
  -- pack_scale_to_amps, F4 SS1 (Z_LOOP_01).
* TestSentinelZeroPointOOne -- the x0.001 factor hits EXACTLY the F4-listed
  non-mass indices (sentinel-per-index construction), not neighbours.
* TestRoundTrip -- pack -> unpack with no physics run (amps_after ==
  packed_before, dens_t=0) conserves every RHOQ_t/CPtot_t/CVtot_t/RHOE_t
  component to ~0 (proven algebraically in the task report, checked here
  numerically).
* TestGaxisVersionStub -- l_gaxis_version in {2, 3} raises NotImplementedError
  in both pack and unpack (F4 SS1.3/SS2.4 v2/v3 -- M5 stubs).
* TestAxisLimitClip -- F4 SS2.4 v1's iag_q/icg_q l_axis_limit clip-overwrite,
  both the triggering (ratio > 1) and non-triggering (ratio <= 1) cases;
  NOT exercised by TestRoundTrip (whose fixture deliberately keeps ratios
  <=1, see _make_scale_raw's comment).
* TestMoistThermoMask -- micptr cloudy mask + thil/qtp diagnosis, F4 SS3.2,
  recomputed independently in-test from F4's quoted formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps, packing
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

NBR = 3
NBI = 2
NBA = 2
NPOINTS = 5


def _make_scale_raw(seed: int = 0, *, nbr=NBR, nbi=NBI, nba=NBA, npoints=NPOINTS):
    """A synthetic, physically-plausible ScaleRawState: mostly-dry air
    (QDRY+QV close to but not exactly 1) with small, strictly positive
    hydrometeor/aerosol loadings -- keeps every intermediate (factor_mxr1,
    moist_denv, mixing ratios) in a sane, strictly-positive range.
    """
    rng = np.random.default_rng(seed)

    dens = rng.uniform(0.5, 1.5, size=npoints)
    qdry = rng.uniform(0.9, 0.999, size=npoints)
    qv = rng.uniform(1.0e-4, 5.0e-3, size=npoints)
    pres = rng.uniform(8.0e4, 1.0e5, size=npoints)
    temp = rng.uniform(250.0, 290.0, size=npoints)
    w = rng.uniform(-1.0, 1.0, size=npoints)
    momz = rng.uniform(-1.0, 1.0, size=npoints)

    ql = rng.uniform(1.0e-6, 1.0e-4, size=(nbr, npoints))
    qi = rng.uniform(1.0e-6, 1.0e-4, size=(nbi, npoints))

    liquid_raw = LiquidState(
        values=rng.uniform(1.0e-7, 1.0e-5, size=(len(LiquidState.PROPS), nbr, 1, npoints))
    )
    ice_raw = IceState(
        values=rng.uniform(1.0e-7, 1.0e-5, size=(len(IceState.PROPS), nbi, 1, npoints))
    )
    # Physical consistency: the a/c-axis-cubed convention (F4 SS4.3 "Cubed-
    # length convention") guarantees ag<=a_len and cg<=c_len, i.e.
    # iag_q<=iacr_q and icg_q<=iccr_q -- unlike every other PPV slot here,
    # iag_q/icg_q are NOT independent of iacr_q/iccr_q, so an unconstrained
    # random draw can (and does) spuriously trip unpack's l_axis_limit clip
    # (F4 SS2.4 v1) even when "no physics ran". Enforce that invariant here
    # so TestRoundTrip probes only pack<->unpack conservation, not the
    # (separately, correctly documented) clip itself.
    ip = index_maps.IcePPV
    ice_raw.values[ip.iag_q.py_idx] = (
        rng.uniform(0.1, 1.0, size=(nbi, 1, npoints)) * (ice_raw.values[ip.iacr_q.py_idx])
    )
    ice_raw.values[ip.icg_q.py_idx] = (
        rng.uniform(0.1, 1.0, size=(nbi, 1, npoints)) * (ice_raw.values[ip.iccr_q.py_idx])
    )
    aerosol_raw = AerosolState(
        values=rng.uniform(1.0e-7, 1.0e-5, size=(len(AerosolState.PROPS), nba, 1, npoints))
    )

    return packing.ScaleRawState(
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


def _prop(thermo: ThermoState, prop: ThermoProp) -> np.ndarray:
    return thermo.values[list(ThermoState.PROPS).index(prop), 0, 0, :]


# ---------------------------------------------------------------------------
# pack_scale_to_amps: thermo block (F4 SS1.1)
# ---------------------------------------------------------------------------


class TestPackThermo:
    def test_factor_mxr1_moist_denv_and_qvv(self):
        scale = _make_scale_raw()
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)

        factor_mxr1 = scale.qdry + scale.qv
        np.testing.assert_allclose(
            _prop(packed.thermo, ThermoProp.moist_denv), scale.dens * factor_mxr1
        )
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.qvv), scale.qv / factor_mxr1)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.ptotv), scale.pres)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.tv), scale.temp)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.wbv), scale.w)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.momv), scale.momz)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.pbv), np.zeros(NPOINTS))

    def test_thv_piv_thetav_formulas(self):
        scale = _make_scale_raw(seed=1)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)

        factor_mxr1 = scale.qdry + scale.qv
        thv_expected = scale.temp * (packing.SCALE_PRE00 / scale.pres) ** (
            packing.SCALE_RDRY / packing.SCALE_CPDRY
        )
        piv_expected = scale.temp / thv_expected * packing.SCALE_CPDRY
        qvv_expected = scale.qv / factor_mxr1
        thetav_expected = thv_expected * (1.0 + 0.61 * qvv_expected)

        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.thv), thv_expected)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.piv), piv_expected)
        np.testing.assert_allclose(_prop(packed.thermo, ThermoProp.thetav), thetav_expected)


# ---------------------------------------------------------------------------
# pack_scale_to_amps: liquid spectrum (F4 SS1.2)
# ---------------------------------------------------------------------------


class TestPackLiquid:
    def test_rmt_q_is_water_plus_aerosol_over_factor_mxr1(self):
        scale = _make_scale_raw(seed=2)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)

        factor_mxr1 = scale.qdry + scale.qv
        lp = index_maps.LiquidPPV
        rmat_raw = scale.liquid_ppv.values[lp.rmat_q.py_idx]  # (nbr,1,npoints)
        expected_rmt = (scale.ql[:, None, :] + rmat_raw) / factor_mxr1
        np.testing.assert_allclose(packed.liquid.values[lp.rmt_q.py_idx], expected_rmt)

    def test_rmat_rmas_plain_conversion(self):
        scale = _make_scale_raw(seed=3)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        lp = index_maps.LiquidPPV
        for idx in (lp.rmat_q.py_idx, lp.rmas_q.py_idx):
            np.testing.assert_allclose(
                packed.liquid.values[idx], scale.liquid_ppv.values[idx] / factor_mxr1
            )

    def test_rcon_q_gets_extra_0p001_division(self):
        scale = _make_scale_raw(seed=4)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        lp = index_maps.LiquidPPV
        expected = scale.liquid_ppv.values[lp.rcon_q.py_idx] / factor_mxr1 / 0.001
        np.testing.assert_allclose(packed.liquid.values[lp.rcon_q.py_idx], expected)


# ---------------------------------------------------------------------------
# pack_scale_to_amps: ice spectrum (F4 SS1.3)
# ---------------------------------------------------------------------------


class TestPackIce:
    def test_imt_q_is_ice_plus_melt_plus_aerosol_over_factor_mxr1(self):
        scale = _make_scale_raw(seed=5)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        ip = index_maps.IcePPV
        imw_raw = scale.ice_ppv.values[ip.imw_q.py_idx]
        imat_raw = scale.ice_ppv.values[ip.imat_q.py_idx]
        expected = (scale.qi[:, None, :] + imw_raw + imat_raw) / factor_mxr1
        np.testing.assert_allclose(packed.ice.values[ip.imt_q.py_idx], expected)

    def test_imw_q_plain_conversion(self):
        scale = _make_scale_raw(seed=6)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        ip = index_maps.IcePPV
        expected = scale.ice_ppv.values[ip.imw_q.py_idx] / factor_mxr1
        np.testing.assert_allclose(packed.ice.values[ip.imw_q.py_idx], expected)

    @pytest.mark.parametrize("name", ["imr_q", "ima_q", "imc_q", "imat_q", "imas_q", "imf_q"])
    def test_mass_props_plain_conversion(self, name):
        scale = _make_scale_raw(seed=7)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        idx = getattr(index_maps.IcePPV, name).py_idx
        expected = scale.ice_ppv.values[idx] / factor_mxr1
        np.testing.assert_allclose(packed.ice.values[idx], expected)

    @pytest.mark.parametrize(
        "name",
        ["icon_q", "ivcs_q", "iacr_q", "iccr_q", "idcr_q", "iag_q", "icg_q", "inex_q"],
    )
    def test_nonmass_props_get_0p001_division(self, name):
        scale = _make_scale_raw(seed=8)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        idx = getattr(index_maps.IcePPV, name).py_idx
        expected = scale.ice_ppv.values[idx] / factor_mxr1 / 0.001
        np.testing.assert_allclose(packed.ice.values[idx], expected)


# ---------------------------------------------------------------------------
# pack_scale_to_amps: aerosol spectrum (F4 SS1.4)
# ---------------------------------------------------------------------------


class TestPackAerosol:
    def test_amt_ams_plain_acon_0p001(self):
        scale = _make_scale_raw(seed=9)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        factor_mxr1 = scale.qdry + scale.qv
        ap = index_maps.AerosolPPV
        for idx in (ap.amt_q.py_idx, ap.ams_q.py_idx):
            np.testing.assert_allclose(
                packed.aerosol.values[idx], scale.aerosol_ppv.values[idx] / factor_mxr1
            )
        np.testing.assert_allclose(
            packed.aerosol.values[ap.acon_q.py_idx],
            scale.aerosol_ppv.values[ap.acon_q.py_idx] / factor_mxr1 / 0.001,
        )


# ---------------------------------------------------------------------------
# pack_scale_to_amps: Emoist bookkeeping (F4 SS1.5), both l_no_ice_heat branches
# ---------------------------------------------------------------------------


class TestPackEmoist:
    def test_emoist_before_without_no_ice_heat(self):
        scale = _make_scale_raw(seed=10)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        expected = -packing.SCALE_LHV0 * scale.qv * scale.dens
        expected = expected + packing.SCALE_LHF0 * np.sum(scale.qi, axis=0) * scale.dens
        np.testing.assert_allclose(packed.emoist_before, expected)
        assert packed.liquid_heat_before is None
        assert packed.ice_heat_before is None

    def test_emoist_before_with_no_ice_heat(self):
        scale = _make_scale_raw(seed=11)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=True)
        expected = -packing.SCALE_LHV0 * scale.qv * scale.dens
        np.testing.assert_allclose(packed.emoist_before, expected)
        assert packed.liquid_heat_before is not None
        assert packed.ice_heat_before is not None
        np.testing.assert_allclose(
            packed.liquid_heat_before, packing.SCALE_LHF0 * np.sum(scale.ql, axis=0) * scale.dens
        )
        np.testing.assert_allclose(
            packed.ice_heat_before, packing.SCALE_LHF0 * np.sum(scale.qi, axis=0) * scale.dens
        )


# ---------------------------------------------------------------------------
# Latent-heat constant anchor: SCALE_LHV0/SCALE_LHS0/SCALE_LHF0 (F4 SS1.5
# Emoist, SS2.7 RHOE_t) against bare numeral literals -- mirrors the
# Exner-constant anchor pattern in test_ref_data.py's TestCaseFromMicroRecord
# .test_field_mapping (its own `rdry, cpdry, pre00 = 287.04, 1004.64, 1.0e5`
# literals). Every OTHER test in this module re-derives its expected value
# from these SAME `packing.SCALE_LH*` module constants, so a silent
# regression in one of them (e.g. a typo'd exponent or digit) would
# otherwise pass every test here -- these anchor the constants themselves
# against independent literals traced to scale_const.F90 (see this module's
# own docstring/comments: SCALE_LHV0 -> scale_const.F90:82, SCALE_LHS0 ->
# scale_const.F90:84, SCALE_LHF0 = SCALE_LHS0 - SCALE_LHV0 ->
# scale_const.F90:206).
# ---------------------------------------------------------------------------


class TestScaleLatentHeatConstantsAnchor:
    def test_scale_lhv0(self):
        assert packing.SCALE_LHV0 == 2.501e6

    def test_scale_lhs0(self):
        assert packing.SCALE_LHS0 == 2.834e6

    def test_scale_lhf0(self):
        assert packing.SCALE_LHF0 == 2.834e6 - 2.501e6


# ---------------------------------------------------------------------------
# Sentinel test: the x0.001 factor hits EXACTLY the F4-listed indices.
# ---------------------------------------------------------------------------


class TestSentinelZeroPointOOne:
    """Every PPV slot set to the SAME sentinel raw value, factor_mxr1 pinned
    to exactly 1 (QDRY=1, QV=0) so plain conversion is a no-op and only the
    x(1/0.001)=x1000 factor distinguishes "mass" from "non-mass" slots.
    """

    SENTINEL = 3.0

    def _scale_with_factor_mxr1_one(self, nbr=NBR, nbi=NBI, nba=NBA, npoints=NPOINTS):
        dens = np.full(npoints, 1.23)
        qdry = np.ones(npoints)
        qv = np.zeros(npoints)
        pres = np.full(npoints, 9.0e4)
        temp = np.full(npoints, 270.0)
        w = np.zeros(npoints)
        momz = np.zeros(npoints)
        ql = np.full((nbr, npoints), self.SENTINEL)
        qi = np.full((nbi, npoints), self.SENTINEL)
        liquid_raw = LiquidState(
            values=np.full((len(LiquidState.PROPS), nbr, 1, npoints), self.SENTINEL)
        )
        ice_raw = IceState(values=np.full((len(IceState.PROPS), nbi, 1, npoints), self.SENTINEL))
        aerosol_raw = AerosolState(
            values=np.full((len(AerosolState.PROPS), nba, 1, npoints), self.SENTINEL)
        )
        return packing.ScaleRawState(
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

    def test_liquid_only_rcon_q_scaled(self):
        scale = self._scale_with_factor_mxr1_one()
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        lp = index_maps.LiquidPPV
        for prop in lp:
            if prop is lp.rmt_q:
                continue  # special: water + aerosol addend, not a plain sentinel echo
            expected = self.SENTINEL * 1000.0 if prop is lp.rcon_q else self.SENTINEL
            np.testing.assert_allclose(
                packed.liquid.values[prop.py_idx], expected, err_msg=f"{prop.name}"
            )

    def test_ice_nonmass_props_scaled_mass_props_not(self):
        scale = self._scale_with_factor_mxr1_one()
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        ip = index_maps.IcePPV
        nonmass = {
            ip.icon_q,
            ip.ivcs_q,
            ip.iacr_q,
            ip.iccr_q,
            ip.idcr_q,
            ip.iag_q,
            ip.icg_q,
            ip.inex_q,
        }
        for prop in ip:
            if prop is ip.imt_q:
                continue  # special: ice + melt + aerosol addend
            expected = self.SENTINEL * 1000.0 if prop in nonmass else self.SENTINEL
            np.testing.assert_allclose(
                packed.ice.values[prop.py_idx], expected, err_msg=f"{prop.name}"
            )

    def test_aerosol_only_acon_q_scaled(self):
        scale = self._scale_with_factor_mxr1_one()
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        ap = index_maps.AerosolPPV
        for prop in ap:
            expected = self.SENTINEL * 1000.0 if prop is ap.acon_q else self.SENTINEL
            np.testing.assert_allclose(
                packed.aerosol.values[prop.py_idx], expected, err_msg=f"{prop.name}"
            )


# ---------------------------------------------------------------------------
# Round trip: pack -> unpack, no physics ran -> every tendency ~ 0.
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("l_no_ice_heat", [False, True])
    def test_zero_tendencies_when_no_physics_ran(self, l_no_ice_heat):
        scale = _make_scale_raw(seed=42)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=l_no_ice_heat)

        dens_t = np.zeros(NPOINTS)
        dt = 30.0

        tend = packing.unpack_amps_to_scale(
            scale,
            packed,
            packed,  # amps_after == packed_before: no physics ran
            dens_t=dens_t,
            dt=dt,
            l_no_ice_heat=l_no_ice_heat,
        )

        atol = 1.0e-9
        np.testing.assert_allclose(tend.dqv, 0.0, atol=atol)
        np.testing.assert_allclose(tend.dql, 0.0, atol=atol)
        np.testing.assert_allclose(tend.dqi, 0.0, atol=atol)
        np.testing.assert_allclose(tend.dqw, 0.0, atol=atol)

        lp = index_maps.LiquidPPV
        for prop in (lp.rmat_q, lp.rmas_q, lp.rcon_q):
            np.testing.assert_allclose(
                tend.d_liquid_ppv.values[prop.py_idx], 0.0, atol=atol, err_msg=prop.name
            )

        ip = index_maps.IcePPV
        for prop in ip:
            if prop in (ip.imt_q, ip.imw_q):
                continue  # unused placeholder slots, see module docstring
            np.testing.assert_allclose(
                tend.d_ice_ppv.values[prop.py_idx], 0.0, atol=atol, err_msg=prop.name
            )

        ap = index_maps.AerosolPPV
        for prop in ap:
            np.testing.assert_allclose(
                tend.d_aerosol_ppv.values[prop.py_idx], 0.0, atol=atol, err_msg=prop.name
            )

        np.testing.assert_allclose(tend.cptot_t, 0.0, atol=atol)
        np.testing.assert_allclose(tend.cvtot_t, 0.0, atol=atol)
        np.testing.assert_allclose(tend.rhoe_t, 0.0, atol=atol)

    def test_nonzero_tendency_when_amps_state_changed(self):
        """Sanity check the round-trip-zero result isn't a tautology from a
        broken formula always returning 0: perturb one AMPS-side value and
        confirm the corresponding tendency becomes nonzero.
        """
        scale = _make_scale_raw(seed=43)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)

        perturbed_liquid_values = packed.liquid.values.copy()
        lp = index_maps.LiquidPPV
        perturbed_liquid_values[lp.rmt_q.py_idx] *= 1.5
        perturbed = packing.PackedAmpsState(
            thermo=packed.thermo,
            liquid=LiquidState(values=perturbed_liquid_values),
            ice=packed.ice,
            aerosol=packed.aerosol,
            emoist_before=packed.emoist_before,
            liquid_heat_before=packed.liquid_heat_before,
            ice_heat_before=packed.ice_heat_before,
        )

        tend = packing.unpack_amps_to_scale(
            scale,
            packed,
            perturbed,
            dens_t=np.zeros(NPOINTS),
            dt=30.0,
            l_no_ice_heat=False,
        )
        assert not np.allclose(tend.dql, 0.0)


# ---------------------------------------------------------------------------
# l_gaxis_version stub: 2/3 raise NotImplementedError (F4 SS1.3/SS2.4, M5).
# ---------------------------------------------------------------------------


class TestGaxisVersionStub:
    @pytest.mark.parametrize("version", [2, 3])
    def test_pack_raises_for_v2_v3(self, version):
        scale = _make_scale_raw(seed=12)
        with pytest.raises(NotImplementedError):
            packing.pack_scale_to_amps(scale, l_no_ice_heat=False, l_gaxis_version=version)

    @pytest.mark.parametrize("version", [2, 3])
    def test_unpack_raises_for_v2_v3(self, version):
        scale = _make_scale_raw(seed=13)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        with pytest.raises(NotImplementedError):
            packing.unpack_amps_to_scale(
                scale,
                packed,
                packed,
                dens_t=np.zeros(NPOINTS),
                dt=30.0,
                l_no_ice_heat=False,
                l_gaxis_version=version,
            )


# ---------------------------------------------------------------------------
# l_axis_limit clip (F4 SS2.4 v1, packing.py:_apply_axis_limit_clip):
# iag_q/icg_q RHOQ_t is overwritten with the a/c-axis value instead of its
# own whenever the AFTER-state axis ratio exceeds 1. TestRoundTrip's fixture
# deliberately keeps ratios <=1 (see _make_scale_raw's comment) so it never
# exercises this branch -- these tests construct ratios >1 (trigger) and
# <=1 (control, confirming no spurious overwrite) directly.
# ---------------------------------------------------------------------------


class TestAxisLimitClip:
    def _scale_and_packed(self, nbi=1, npoints=3, seed=100):
        scale = _make_scale_raw(seed=seed, nbi=nbi, npoints=npoints)
        packed = packing.pack_scale_to_amps(scale, l_no_ice_heat=False)
        return scale, packed

    def _amps_after_with_cog(self, packed, iag_values, icg_values):
        ip = index_maps.IcePPV
        values = packed.ice.values.copy()
        values[ip.iag_q.py_idx] = iag_values
        values[ip.icg_q.py_idx] = icg_values
        return packing.PackedAmpsState(
            thermo=packed.thermo,
            liquid=packed.liquid,
            ice=IceState(values=values),
            aerosol=packed.aerosol,
            emoist_before=packed.emoist_before,
            liquid_heat_before=packed.liquid_heat_before,
            ice_heat_before=packed.ice_heat_before,
        )

    def test_clip_triggers_for_both_iag_and_icg_when_ratio_exceeds_one(self):
        scale, packed = self._scale_and_packed()
        ip = index_maps.IcePPV
        dt = 20.0

        iacr_after = packed.ice.values[ip.iacr_q.py_idx]
        iccr_after = packed.ice.values[ip.iccr_q.py_idx]
        # Force both ratios strictly > 1 (2x / 3x the axis value).
        amps_after = self._amps_after_with_cog(packed, iacr_after * 2.0, iccr_after * 3.0)

        tend = packing.unpack_amps_to_scale(
            scale,
            packed,
            amps_after,
            dens_t=np.zeros_like(scale.dens),
            dt=dt,
            l_no_ice_heat=False,
        )

        factor_mxr1 = scale.qdry + scale.qv
        moist_denv_after = scale.dens * factor_mxr1
        dens = scale.dens
        iag_raw = scale.ice_ppv.values[ip.iag_q.py_idx]
        icg_raw = scale.ice_ppv.values[ip.icg_q.py_idx]

        # Hand-computed clipped result: cross-referencing iacr_q's/iccr_q's
        # OWN after-value into iag_q's/icg_q's tendency (F4 SS2.4 v1,
        # packing.py:430-452), NOT iag_q's/icg_q's own after-value.
        expected_iag = (iacr_after * moist_denv_after * 0.001 - iag_raw * dens) / dt
        expected_icg = (iccr_after * moist_denv_after * 0.001 - icg_raw * dens) / dt

        np.testing.assert_allclose(tend.d_ice_ppv.values[ip.iag_q.py_idx], expected_iag, atol=1e-9)
        np.testing.assert_allclose(tend.d_ice_ppv.values[ip.icg_q.py_idx], expected_icg, atol=1e-9)

        # Sanity: the clipped result must differ from the (wrong) plain
        # formula that would use iag_q's/icg_q's own after-value -- if a
        # future edit swapped axis_after/cog_after or flipped the mask/ratio
        # comparison, this catches it (the clip would silently no-op).
        plain_iag = (iacr_after * 2.0 * moist_denv_after * 0.001 - iag_raw * dens) / dt
        assert not np.allclose(tend.d_ice_ppv.values[ip.iag_q.py_idx], plain_iag)

    def test_no_clip_when_ratio_is_at_most_one(self):
        scale, packed = self._scale_and_packed(seed=101)
        ip = index_maps.IcePPV
        dt = 20.0

        iacr_after = packed.ice.values[ip.iacr_q.py_idx]
        iccr_after = packed.ice.values[ip.iccr_q.py_idx]
        iag_after = iacr_after * 0.5  # ratio 0.5 <= 1: must NOT trigger the clip
        icg_after = iccr_after * 0.5
        amps_after = self._amps_after_with_cog(packed, iag_after, icg_after)

        tend = packing.unpack_amps_to_scale(
            scale,
            packed,
            amps_after,
            dens_t=np.zeros_like(scale.dens),
            dt=dt,
            l_no_ice_heat=False,
        )

        factor_mxr1 = scale.qdry + scale.qv
        moist_denv_after = scale.dens * factor_mxr1
        dens = scale.dens
        iag_raw = scale.ice_ppv.values[ip.iag_q.py_idx]
        icg_raw = scale.ice_ppv.values[ip.icg_q.py_idx]

        # Plain (unclipped) RHOQ_t formula, using iag_q's/icg_q's OWN
        # after-value -- i.e. confirming NO overwrite happened.
        expected_iag = (iag_after * moist_denv_after * 0.001 - iag_raw * dens) / dt
        expected_icg = (icg_after * moist_denv_after * 0.001 - icg_raw * dens) / dt

        np.testing.assert_allclose(tend.d_ice_ppv.values[ip.iag_q.py_idx], expected_iag, atol=1e-9)
        np.testing.assert_allclose(tend.d_ice_ppv.values[ip.icg_q.py_idx], expected_icg, atol=1e-9)


# ---------------------------------------------------------------------------
# moistthermo_mask: cloudy mask + thil/qtp diagnosis (F4 SS3.2).
# ---------------------------------------------------------------------------


class TestMoistThermoMask:
    def _make_liquid_ice(self, nbr=4, nbi=3, npoints=2, seed=0):
        rng = np.random.default_rng(seed)
        liquid = LiquidState(
            values=rng.uniform(1.0e-25, 1.0e-4, size=(len(LiquidState.PROPS), nbr, 1, npoints))
        )
        ice = IceState(
            values=rng.uniform(1.0e-25, 1.0e-4, size=(len(IceState.PROPS), nbi, 1, npoints))
        )
        return liquid, ice

    def test_thil_qtp_match_f4_formula(self):
        """Recompute thp/qtp independently, from F4's own quoted formula,
        using hand-picked qc/qr/qi/qv/th/t (bypassing the bin-threshold
        logic entirely, isolating the closed-form diagnosis)."""
        liquid, ice = self._make_liquid_ice(nbr=1, nbi=1, npoints=3, seed=1)
        # Force every bin comfortably above threshold so qc/qr/qi are just
        # the bins' own (mass - aerosol[- melt]) values, deterministically.
        lp, ip = index_maps.LiquidPPV, index_maps.IcePPV
        liquid.values[lp.rmt_q.py_idx] = np.array([[[2.0e-3, 3.0e-3, 1.0e-3]]])
        liquid.values[lp.rmat_q.py_idx] = np.array([[[0.2e-3, 0.3e-3, 0.1e-3]]])
        ice.values[ip.imt_q.py_idx] = np.array([[[4.0e-3, 1.0e-3, 5.0e-3]]])
        ice.values[ip.imat_q.py_idx] = np.array([[[0.4e-3, 0.1e-3, 0.5e-3]]])
        ice.values[ip.imw_q.py_idx] = np.array([[[0.1e-3, 0.05e-3, 0.2e-3]]])

        qv = np.array([1.0e-3, 2.0e-3, 0.5e-3])
        th = np.array([300.0, 295.0, 305.0])
        t = np.array([270.0, 260.0, 280.0])

        result = packing.moistthermo_mask(liquid, ice, qv, th, t, nbhzcl=0)

        # With nbhzcl=0, ibr_st=1, so the "cloud" loop (ibr=1..ibr_st-1) is
        # empty (Fortran do 1,0 -> 0 iterations) -- qc(k) is identically 0,
        # and ALL liquid mass lands in qr instead.
        qr_liquid_expected = (
            liquid.values[lp.rmt_q.py_idx, 0, 0, :] - liquid.values[lp.rmat_q.py_idx, 0, 0, :]
        )
        qi_expected = (
            ice.values[ip.imt_q.py_idx, 0, 0, :]
            - ice.values[ip.imat_q.py_idx, 0, 0, :]
            - ice.values[ip.imw_q.py_idx, 0, 0, :]
        )
        qr_ice_expected = ice.values[ip.imw_q.py_idx, 0, 0, :]
        qr_expected = qr_liquid_expected + qr_ice_expected
        qc_expected = np.zeros_like(qr_expected)

        aklv = packing.SCALE_LHV0 / packing.SCALE_CPDRY
        akiv = packing.SCALE_LHS0 / packing.SCALE_CPDRY
        thp_expected = th / (
            1.0 + (aklv * (qr_expected + qc_expected) + akiv * qi_expected) / np.maximum(t, 253.0)
        )
        # qtotal's own accumulation (F4 SS3.2) is a SEPARATE running total
        # from qc/qr/qi: its liquid term is (rmt-rmat) and its ice term is
        # (imt-imat) -- NOT (imt-imat-imw) like qi(k) -- i.e. melt water is
        # folded into qtotal via the ice term, not double-counted via a
        # separate qr-like term. Transcribed verbatim, not qr_expected+qi_expected+qv.
        qtotal_liquid_term = (
            liquid.values[lp.rmt_q.py_idx, 0, 0, :] - liquid.values[lp.rmat_q.py_idx, 0, 0, :]
        )
        qtotal_ice_term = (
            ice.values[ip.imt_q.py_idx, 0, 0, :] - ice.values[ip.imat_q.py_idx, 0, 0, :]
        )
        qtp_expected = qtotal_liquid_term + qtotal_ice_term + qv

        np.testing.assert_allclose(result.qc, qc_expected, atol=1e-12)
        np.testing.assert_allclose(result.qr, qr_expected, atol=1e-12)
        np.testing.assert_allclose(result.qi, qi_expected, atol=1e-12)
        np.testing.assert_allclose(result.qtp, qtp_expected, atol=1e-12)
        np.testing.assert_allclose(result.thp, thp_expected, atol=1e-12)

    def test_subthreshold_bins_are_zeroed_and_do_not_set_micptr(self):
        liquid, ice = self._make_liquid_ice(nbr=2, nbi=2, npoints=1, seed=2)
        lp = index_maps.LiquidPPV
        # Bin 0: below RRLMTB (all-zero contribution + zeroed in place).
        liquid.values[:, 0, 0, :] = 1.0e-30
        liquid.values[lp.rmt_q.py_idx, 0, 0, :] = 1.0e-30
        # Bin 1: above RRLMTB.
        liquid.values[lp.rmt_q.py_idx, 1, 0, :] = 5.0e-4
        liquid.values[lp.rmat_q.py_idx, 1, 0, :] = 1.0e-4

        ice.values[:, :, 0, :] = 1.0e-30  # both ice bins below RILMTB

        qv = np.array([1.0e-4])
        th = np.array([300.0])
        t = np.array([270.0])

        result = packing.moistthermo_mask(liquid, ice, qv, th, t, nbhzcl=0)

        # Sub-threshold liquid bin 0 fully zeroed in the returned state.
        np.testing.assert_allclose(result.liquid.values[:, 0, 0, :], 0.0)
        # Sub-threshold ice bins fully zeroed.
        np.testing.assert_allclose(result.ice.values, 0.0)
        # micptr set (bin 1 passed).
        assert result.micptr[0] == 1

    def test_all_bins_subthreshold_gives_micptr_zero(self):
        liquid, ice = self._make_liquid_ice(nbr=2, nbi=2, npoints=1, seed=3)
        liquid.values[:] = 1.0e-30
        ice.values[:] = 1.0e-30
        qv = np.array([1.0e-4])
        th = np.array([300.0])
        t = np.array([270.0])

        result = packing.moistthermo_mask(liquid, ice, qv, th, t, nbhzcl=0)
        assert result.micptr[0] == 0
        np.testing.assert_allclose(result.qc, 0.0)
        np.testing.assert_allclose(result.qr, 0.0)
        np.testing.assert_allclose(result.qi, 0.0)

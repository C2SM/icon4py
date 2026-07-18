# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/liquid_diag.py (M2a Task 2): `diag_pq`'s liquid branch +
liquid terminal velocity, per
docs/superpowers/facts/m2/vapor-deposition.md ("G3") §5 and
docs/superpowers/facts/m2/sedimentation-terminalvel.md ("G5").

`TestGoldenBin` is the primary correctness check: `_golden_bin` is a
scalar (pure `math`, no numpy, no import of `liquid_diag` internals)
transcription of G3 §5 + the five helper routines it calls
(`cal_meanmass_vec`, `cal_den_aclen_vec`, `cal_terminal_vel_vec`,
`cal_ventilation_coef_vec`, `cal_capacitance_vec`, `cal_coef_vapdep2_vec`),
independently re-derived from the same Fortran read for this task (see
`core/liquid_diag.py`'s module docstring for file:line anchors) -- NOT a
copy-paste of the module-under-test's numpy code. It is exercised at three
drop sizes spanning `cal_terminal_vel_vec`'s three nontrivial regimes
(Stokes / empirical-fit / Bond-number) plus one aerosol-bearing bin, each
compared against `diag_pq_liquid`'s batched output for the corresponding
bin.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps, liquid_diag, thermo
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
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


def _liquid_state(bins: list[tuple[float, float, float, float]]) -> LiquidState:
    """`bins`: list of (rmt, rcon, rmat, rmas) per bin, single column."""
    lp = index_maps.LiquidPPV
    nbins = len(bins)
    values = np.zeros((len(LiquidState.PROPS), nbins, 1, 1), dtype=np.float64)
    for b, (rmt, rcon, rmat, rmas) in enumerate(bins):
        values[lp.rmt_q.py_idx, b, 0, 0] = rmt
        values[lp.rcon_q.py_idx, b, 0, 0] = rcon
        values[lp.rmat_q.py_idx, b, 0, 0] = rmat
        values[lp.rmas_q.py_idx, b, 0, 0] = rmas
    return LiquidState(values=values)


P_STD = float(AmpsConst.p00)  # 1e6 dyn/cm^2 ~ 1000 hPa
T_STD = 280.0  # K
DEN_STD = 1.2e-3  # g/cm^3, moist air
QV_STD = 1.0e-2  # dimensionless


def _drop_mean_mass(radius_cm: float, density: float = 1.0) -> float:
    """mean_mass for a pure-water sphere of the given radius, den_w=1."""
    diameter = 2.0 * radius_cm
    return (math.pi / 6.0) * diameter**3 * density


# ---------------------------------------------------------------------------
# Golden scalar reference -- independent transcription of G3 §5 + helpers.
# ---------------------------------------------------------------------------


def _golden_bin(
    *,
    rmt: float,
    rcon: float,
    rmat: float,
    rmas: float,
    t: float,
    p: float,
    den: float,
    qv: float,
    config: AmpsConfig,
) -> dict[str, float]:
    """Scalar reference for diag_pq's liquid branch on one bin. Returns a
    dict with the same keys as LiquidDiag's fields (mean_mass, length,
    a_len, c_len, density, terminal_velocity, capacitance, ventilation_fv,
    ventilation_fh, ventilation_fkn, vapdep_coef1, vapdep_coef2)."""
    active = rcon > 1.0e-30 and rmt > 1.0e-30
    mean_mass = rmt / rcon if active else 0.0
    if mean_mass == 0.0:
        active = False

    if not active:
        return {
            "mean_mass": 0.0,
            "length": 0.0,
            "a_len": 0.0,
            "c_len": 0.0,
            "density": 1.0,
            "terminal_velocity": 0.0,
            "capacitance": 0.0,
            "ventilation_fv": 1.0,
            "ventilation_fh": 1.0,
            "ventilation_fkn": 1.0,
            "vapdep_coef1": 0.0,
            "vapdep_coef2": 0.0,
        }

    den_w = float(AmpsConst.den_w)
    pi = float(AmpsConst.PI)
    coef4pi3 = float(AmpsConst.coef4pi3)
    gg = float(AmpsConst.gg)
    r_v = float(AmpsConst.R_v)
    m_w = float(AmpsConst.M_w)
    a_cliq = float(AmpsConst.a_cliq)
    l_e = float(AmpsConst.L_e)
    c_pa = float(AmpsConst.C_pa)

    eps_map = min(1.0, max(0.0, rmas / rmat)) if rmat > 0.0 else config.eps_ap[0]

    den_as = config.den_aps[0]
    den_ai = config.den_api[0]
    den_ap = den_ai / (1.0 - eps_map * (1.0 - den_ai / den_as))

    map_ = rmat / rcon
    density = mean_mass / ((mean_mass - map_) / den_w + map_ / den_ap)
    r_n = (map_ / coef4pi3 / den_ap) ** (1.0 / 3.0)
    length = (6.0 * (mean_mass / (pi * density))) ** (1.0 / 3.0)
    length = max(r_n * 1.05, length)
    density = mean_mass / (pi / 6.0 * length**3)

    if length <= 280.0e-4:
        a_len = length / 2.0
        c_len = length / 2.0
    elif length <= 1.0e-1:
        dum = length
        alpha = max(
            1.001668 - 0.098055 * dum - 2.52686 * dum**2 + 3.75061 * dum**3 - 1.68692 * dum**4,
            1.0e-5,
        )
        dum2 = length / (8.0 * alpha) ** (1.0 / 3.0)
        a_len = dum2
        c_len = dum2 * alpha
    else:
        dum = min(0.9, length)
        alpha = max(
            1.001668 - 0.098055 * dum - 2.52686 * dum**2 + 3.75061 * dum**3 - 1.68692 * dum**4,
            1.0e-5,
        )
        dum2 = length / (8.0 * alpha) ** (1.0 / 3.0)
        a_len = dum2
        c_len = dum2 * alpha

    den_a = den * (1.0 - qv)
    d_vis = float(thermo.dynamic_viscosity(t))
    sig_wa = float(thermo.sfc_tension(t))
    d_v = float(thermo.diffusivity(p, t))
    k_a = float(thermo.thermal_conductivity(t))

    rad = length * 0.5
    if rad < 0.5e-4:
        vtm = 0.0
    elif rad < 10.0e-4:
        u_s = rad**2 * gg * (den_w - den_a) / 4.5 / d_vis
        lambda_a = 6.6e-6 * (d_vis / 1.818e-4) * (1013250.0 / p) * (t / 293.15)
        vtm = (1.0 + 1.26 * lambda_a / rad) * u_s
    elif rad < 535.0e-4:
        x = math.log(32.0 * rad**3 * (den_w - den_a) * den_a * gg / 3.0 / d_vis**2)
        y = (
            -0.318657e1
            + 0.992696 * x
            - 0.153193e-2 * x * x
            - 0.987059e-3 * x * x * x
            - 0.578878e-3 * x**4
            + 0.855176e-4 * x**5
            - 0.327815e-5 * x**6
        )
        nre = math.exp(y)
        vtm = nre * d_vis / (2.0 * rad * den_a)
    else:
        rad = min(rad, 3500.0e-4)
        nbo = gg * (den_w - den_a) * rad**2 / sig_wa
        np_ = sig_wa**3 * den_a**2 / d_vis**4 / gg
        x = math.log(nbo * np_ ** (1.0 / 6.0) * 16.0 / 3.0)
        y = (
            -0.500015e1
            + 0.523778e1 * x
            - 0.204914e1 * x * x
            + 0.475294 * x**3
            - 0.542819e-1 * x**4
            + 0.238449e-2 * x**5
        )
        nre = np_ ** (1.0 / 6.0) * math.exp(y)
        vtm = nre * d_vis / (2.0 * rad * den_a)

    cap = 0.5 * length

    n_sc = d_vis / den / d_v
    n_ns = d_vis / den / k_a
    nre_bin = length * vtm * den / d_vis
    x_v = (n_sc ** (1.0 / 3.0)) * math.sqrt(nre_bin)
    x_h = (n_ns ** (1.0 / 3.0)) * math.sqrt(nre_bin)

    def _vent(x: float) -> float:
        if x < 1.4:
            return 1.0 + 0.108 * x**2
        elif x <= 51.4:
            return 0.78 + 0.308 * x
        return 0.78 + 0.308 * 51.4

    fv = _vent(x_v)
    fh = _vent(x_h)

    r_m = length * 0.5
    beta_w = 0.036
    delta = 1.0e-5
    fkn = 1.0 / (r_m / (r_m + delta) + (d_v / beta_w / r_m) * math.sqrt(2.0 * pi / (r_v * t)))

    aa = 2.0 * sig_wa / (r_v * t * den_w)
    nu_aps0 = config.nu_aps[0]
    m_aps0 = config.M_aps[0]
    phi_aps0 = config.phi_aps[0]
    sb = nu_aps0 * eps_map * m_w * den_ap / (m_aps0 * den_w) * phi_aps0
    beta = 0.5
    r_n3 = r_n**3
    s_salt = aa / a_len - sb * r_n ** (2.0 * (1.0 + beta)) / (a_len**3 - r_n3)

    vw = math.sqrt(8.0 / pi * r_v * t)
    buzai_con = 4.0 * d_v / (vw * a_cliq)

    estbar, esitbar = thermo.make_esat_tables()
    e_sat1 = float(thermo.esat_lk(1, t, estbar, esitbar))
    rho_s = e_sat1 / (t * r_v)
    gamma_w = 1.0 + l_e * rho_s / (c_pa * t * den) * (l_e / r_v / t - 1.0)

    coef1 = 4.0 * pi * d_v * rho_s / gamma_w * cap * cap / (cap + buzai_con) * fv
    coef2 = -coef1 * s_salt

    return {
        "mean_mass": mean_mass,
        "length": length,
        "a_len": a_len,
        "c_len": c_len,
        "density": density,
        "terminal_velocity": vtm,
        "capacitance": cap,
        "ventilation_fv": fv,
        "ventilation_fh": fh,
        "ventilation_fkn": fkn,
        "vapdep_coef1": coef1,
        "vapdep_coef2": coef2,
    }


def _diag_field(diag: liquid_diag.LiquidDiag, key: str, bin_idx: int) -> float:
    return float(getattr(diag, key)[bin_idx, 0])


class TestGoldenBin:
    """Compare diag_pq_liquid's batched output against `_golden_bin`'s
    independent scalar transcription, at three radii spanning
    cal_terminal_vel_vec's three nontrivial regimes, plus one aerosol-
    bearing bin."""

    RADII_CM = {
        "stokes": 2.0e-4,  # 2 um radius: Stokes regime (0.5e-4 <= rad < 10e-4)
        "mid": 200.0e-4,  # 200 um radius: empirical-fit regime (10e-4 <= rad < 535e-4)
        "large": 700.0e-4,  # 700 um radius: Bond-number regime (rad >= 535e-4)
    }

    @pytest.fixture(scope="class")
    def config(self) -> AmpsConfig:
        return AmpsConfig.cloudlab()

    @pytest.mark.parametrize("regime", ["stokes", "mid", "large"])
    def test_pure_water_bin(self, config, luts, regime):
        radius = self.RADII_CM[regime]
        mean_mass = _drop_mean_mass(radius)
        rcon = 1.0
        rmt = mean_mass * rcon

        golden = _golden_bin(
            rmt=rmt,
            rcon=rcon,
            rmat=0.0,
            rmas=0.0,
            t=T_STD,
            p=P_STD,
            den=DEN_STD,
            qv=QV_STD,
            config=config,
        )
        liquid = _liquid_state([(rmt, rcon, 0.0, 0.0)])
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)
        diag = liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts)

        for key, expected in golden.items():
            got = _diag_field(diag, key, 0)
            assert got == pytest.approx(expected, rel=1e-10), key

        # Pure drop (no aerosol): density recovers den_w exactly.
        assert _diag_field(diag, "density", 0) == pytest.approx(float(AmpsConst.den_w), rel=1e-10)
        # Capacitance is always 0.5*length for liquid (sphere assumption).
        assert _diag_field(diag, "capacitance", 0) == pytest.approx(
            0.5 * _diag_field(diag, "length", 0), rel=1e-12
        )

    def test_aerosol_bearing_bin(self, config, luts):
        radius = self.RADII_CM["mid"]
        mean_mass = _drop_mean_mass(radius)
        rcon = 1.0
        rmt = mean_mass * rcon
        rmat = 0.01 * rmt  # 1% of mass is aerosol
        rmas = rmat  # fully soluble (matches cloudlab eps_ap[0]=1.0)

        golden = _golden_bin(
            rmt=rmt,
            rcon=rcon,
            rmat=rmat,
            rmas=rmas,
            t=T_STD,
            p=P_STD,
            den=DEN_STD,
            qv=QV_STD,
            config=config,
        )
        liquid = _liquid_state([(rmt, rcon, rmat, rmas)])
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)
        diag = liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts)

        for key, expected in golden.items():
            got = _diag_field(diag, key, 0)
            assert got == pytest.approx(expected, rel=1e-9), key


# ---------------------------------------------------------------------------
# Mean mass: bin-mass-weighted (cal_meanmass_vec: mean_mass = mass(1)/con).
# ---------------------------------------------------------------------------


class TestMeanMass:
    def test_mass_weighted(self, luts):
        config = AmpsConfig.cloudlab()
        rcon, rmt = 3.0, 9.0e-9
        liquid = _liquid_state([(rmt, rcon, 0.0, 0.0)])
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)
        diag = liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts)
        assert _diag_field(diag, "mean_mass", 0) == pytest.approx(rmt / rcon, rel=1e-14)

    def test_multi_bin_independence(self, luts):
        """Each bin's mean_mass depends only on its own (rmt, rcon)."""
        config = AmpsConfig.cloudlab()
        bins = [(9.0e-9, 3.0, 0.0, 0.0), (4.0e-8, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
        liquid = _liquid_state(bins)
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)
        diag = liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts)
        assert _diag_field(diag, "mean_mass", 0) == pytest.approx(9.0e-9 / 3.0, rel=1e-14)
        assert _diag_field(diag, "mean_mass", 1) == pytest.approx(4.0e-8 / 2.0, rel=1e-14)
        assert _diag_field(diag, "mean_mass", 2) == 0.0


# ---------------------------------------------------------------------------
# Inactive-bin (icond1==1) defaults -- diag_pq's own pre-phase-branch init.
# ---------------------------------------------------------------------------


class TestInactiveBinDefaults:
    def test_zero_mass_and_con(self, luts):
        config = AmpsConfig.cloudlab()
        liquid = _liquid_state([(0.0, 0.0, 0.0, 0.0)])
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)
        diag = liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts)

        assert _diag_field(diag, "mean_mass", 0) == 0.0
        assert _diag_field(diag, "length", 0) == 0.0
        assert _diag_field(diag, "a_len", 0) == 0.0
        assert _diag_field(diag, "c_len", 0) == 0.0
        assert _diag_field(diag, "density", 0) == 1.0
        assert _diag_field(diag, "terminal_velocity", 0) == 0.0
        assert _diag_field(diag, "capacitance", 0) == 0.0
        assert _diag_field(diag, "ventilation_fv", 0) == 1.0
        assert _diag_field(diag, "ventilation_fh", 0) == 1.0
        assert _diag_field(diag, "ventilation_fkn", 0) == 1.0
        assert _diag_field(diag, "vapdep_coef1", 0) == 0.0
        assert _diag_field(diag, "vapdep_coef2", 0) == 0.0

    def test_below_threshold_con_or_mass(self, luts):
        """con or mass(1) <= 1e-30 alone (not both zero) is still inactive."""
        config = AmpsConfig.cloudlab()
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)

        liquid_no_con = _liquid_state([(1.0e-9, 1.0e-31, 0.0, 0.0)])
        diag = liquid_diag.diag_pq_liquid(liquid_no_con, thermo_state, config, luts)
        assert _diag_field(diag, "density", 0) == 1.0

        liquid_no_mass = _liquid_state([(1.0e-31, 1.0, 0.0, 0.0)])
        diag = liquid_diag.diag_pq_liquid(liquid_no_mass, thermo_state, config, luts)
        assert _diag_field(diag, "density", 0) == 1.0


# ---------------------------------------------------------------------------
# Shape.
# ---------------------------------------------------------------------------


class TestShape:
    def test_output_shape_matches_input(self, luts):
        config = AmpsConfig.cloudlab()
        bins = [(9.0e-9, 3.0, 0.0, 0.0) for _ in range(5)]
        liquid = _liquid_state(bins)
        thermo_state = _thermo_state(p=P_STD, t=T_STD, den=DEN_STD, qv=QV_STD)
        diag = liquid_diag.diag_pq_liquid(liquid, thermo_state, config, luts)
        for f in dataclasses.fields(diag):
            arr = getattr(diag, f.name)
            assert arr.shape == (5, 1)
            assert arr.dtype == np.float64

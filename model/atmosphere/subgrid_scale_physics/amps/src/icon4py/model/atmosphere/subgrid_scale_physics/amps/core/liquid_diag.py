# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`diag_pq`'s LIQUID branch + liquid terminal velocity, transcribed
verbatim from AMPS Fortran (scale_amps repo), per
docs/superpowers/facts/m2/vapor-deposition.md ("G3" below) §5 (`diag_pq`,
`mod_amps_core.F90:12552`, the `if(phase==1)` branch, lines 12786-12835) and
docs/superpowers/facts/m2/sedimentation-terminalvel.md ("G5" below).

`diag_pq`'s liquid branch calls FIVE helpers, none of which G3 quotes in
full body (G3 names them and quotes only their call sites/summary table);
per this task's dispatch authorization ("reading the named Fortran routine
directly, quoting into report" when a named routine is truncated/not
quoted), all five were read directly from `contrib/AMPS/` and are
transcribed verbatim below, each in its own section:

* `cal_meanmass_vec` (`class_Mass_Bin.F90:1840-1849`) -- mean_mass =
  mass(1)/con.
* `cal_den_aclen_vec` (`class_Group.F90:11222-11300`) -- bulk density +
  equivalent diameter (`len`) + spheroid semi-axes (`a_len`/`c_len`).
* `cal_terminal_vel_vec`, phase==1 branch only (`class_Group.F90:8069-8119`)
  -- per-bin (mean-mass-point) fall speed `vtm`: Stokes / empirical-fit /
  Bond-number regimes (Beard 1976-style), NOT the bin-integrated
  Best-number/Böhm fit `cal_wterm_vel_v3_vec` G5 documents (that is a
  DIFFERENT function, used by the sedimentation driver `w_terminal_vel`,
  not by `diag_pq`) -- G5's own reference to "`w_terminal_vel` /
  `cal_terminal_vel_vec`" bundles both under one topic heading, but only
  `cal_terminal_vel_vec` is on `diag_pq`'s actual call path (`diag_pq`
  calls `cal_terminal_vel_vec(phase,g,ag,icond1,1)` directly, G3 §5's own
  quoted call list) and is the routine this module ports.
* `cal_ventilation_coef_vec`, phase==1 branch (G3 §3b, quoted in full).
* `cal_capacitance_vec`, phase==1 branch (G3 §3a, quoted in full):
  `CAP = 0.5*len` (sphere assumption).
* `cal_coef_vapdep2_vec`, phase==1 branch (G3 §3c, quoted in full):
  Khvorostyanov & Curry (2014) Köhler-based growth coefficients
  `coef(1)`/`coef(2)`.
* `get_fkn` (`class_Thermo_Var.F90:707-730`) -- kinetic correction factor,
  called from `cal_ventilation_coef_vec`.

Scope, matching this task's Deliverable field list (mean_mass, length,
density, terminal_velocity, capacitance, ventilation coefficient,
vapor-deposition coefficient) -- two `diag_pq` phase==1 outputs are
DELIBERATELY NOT exposed on `LiquidDiag`:

* `eps_map` (soluble aerosol mass fraction) and `r_n`/`r_n3` (dry aerosol
  radius) are computed internally (needed by `cal_den_aclen_vec` and
  `cal_coef_vapdep2_vec`) but not returned -- not in the task's named field
  list.
* `r_crt`/`r_act` (critical/haze radius, `get_critrad_anal`/
  `get_hazerad_anal`) -- also computed by `cal_coef_vapdep2_vec` but by TWO
  further un-quoted helper functions not read for this task (out of the
  named field list; used elsewhere for the haze-transfer evaporation check
  in `vapor_deposition`, not by anything in this module's scope).
* `cal_surface_temp2_vec` (droplet surface temperature, called LAST in
  G3 §5's liquid branch, after the vapor-dep coefficients) -- not quoted in
  G3, not in the task's named field list, skipped.

Aerosol-category indexing: `cal_coef_vapdep2_vec`'s liquid branch hardcodes
`nu_aps(1)`, `M_aps(1)`, `phi_aps(1)` (Fortran 1-based, i.e. category index
0 here) -- NOT a per-bin category lookup. `cal_den_aclen_vec`'s
`g%MS(i,n)%den_as`/`den_ai` (per-bin aerosol-material densities) are, in
turn, initialized group-wide from `den_aps0(ica)`/`den_api0(ica)`
(`class_Group.F90:1598-1599`) at group-creation time -- read directly to
confirm the SAME single category index drives both, so this module
consistently uses `config.{nu_aps,M_aps,phi_aps,den_aps,den_api,eps_ap}[0]`
throughout (category 1, matching `cal_coef_vapdep2_vec`'s own hardcoded
index -- not an approximation). This is moot for cloudlab specifically
(`den_aps`/`den_api`/`nu_aps`/`phi_aps` are identical across all 4
categories there), but the [0]-indexing choice is independently justified
by the Fortran's own hardcoding, not just cloudlab's degenerate config.

Dry-air density (`ag%TV(n)%den_a`, needed only by `cal_terminal_vel_vec`'s
Stokes/empirical-fit branches): the Fortran computes it as
`M_a*(P-e)/(R_u*T)` (`e` = actual, not saturation, vapor pressure) via a
`Make_Thermo_Var`/`AirGroup` construction this port's M1 `ThermoState` does
not carry (`ThermoState.moist_denv` is `state.py`'s pass-through of the
driving model's own moist-air density field, per `core/packing.py`'s
`pack_scale_to_amps`: `moist_denv = dens * factor_mxr1` -- NOT computed via
that ideal-gas formula at all). Rather than reconstruct `e` from an
R_v-based approximation, this module uses the EXACT mass-conservation
identity `den_a = den * (1 - qv)` (`qv` = vapor mass per unit moist-air
mass, `ThermoProp.qvv`; dry-air mass = moist-air mass minus vapor mass, by
definition of a mixing ratio) -- a documented, physically-exact-given-its-
own-definitions substitute for the Fortran's `e`-based formula, not a
guessed approximation. See the task report for this call-out.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import thermo as thermo_fn
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import AmpsLuts
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# LiquidDiag -- diag_pq_liquid's return bundle. All arrays (nbins, npoints)
# float64, one entry per (bin, column) -- see module docstring for exactly
# which G3 §5 outputs are/aren't exposed.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LiquidDiag:
    """Per-bin liquid diagnostics from `diag_pq`'s phase==1 branch (G3 §5)
    plus `cal_terminal_vel_vec`'s phase==1 branch (G5). All fields
    `(nbins, npoints)` float64, aligned with the `LiquidState` they were
    computed from.

    Bins with no water (`con<=1e-30` or `mass(1)<=1e-30`, or a
    `mean_mass` that computes to exactly 0.0 -- G3 §5's `icond1`/`em`
    guard) carry the Fortran `diag_pq`'s own pre-phase-branch INIT
    defaults, not zeros-by-omission: `density=1.0`, `terminal_velocity=0.0`,
    `capacitance=0.0`, `ventilation_fv=ventilation_fh=ventilation_fkn=1.0`,
    `vapdep_coef1=vapdep_coef2=0.0`, `mean_mass=length=a_len=c_len=0.0`
    (the last two are NOT explicitly re-initialized in the Fortran itself
    for a masked-out bin -- see module docstring's `a_len`/`c_len` note --
    0.0 is this module's own, documented, stateless-diag choice).
    """

    mean_mass: np.ndarray  # g; g%MS%mean_mass
    length: np.ndarray  # cm; g%MS%len (equivalent-sphere diameter)
    a_len: np.ndarray  # cm; g%MS%a_len (spheroid semi-major/equatorial axis)
    c_len: np.ndarray  # cm; g%MS%c_len (spheroid semi-minor/polar axis)
    density: np.ndarray  # g/cm^3; g%MS%den
    terminal_velocity: np.ndarray  # cm/s; g%MS%vtm
    capacitance: np.ndarray  # cm; g%MS%CAP
    ventilation_fv: np.ndarray  # g%MS%fv (vapor ventilation coefficient)
    ventilation_fh: np.ndarray  # g%MS%fh (heat ventilation coefficient)
    ventilation_fkn: np.ndarray  # g%MS%fkn (kinetic correction factor)
    vapdep_coef1: np.ndarray  # g%MS%coef(1); vapor-deposition growth coeff
    vapdep_coef2: np.ndarray  # g%MS%coef(2); vapor-deposition growth coeff


# ---------------------------------------------------------------------------
# Small numeric helpers -- safe division / suppressing the expected
# divide-by-zero warnings for masked-out (icond1==1) bins, whose computed
# values are discarded by the final np.where(active, ...) regardless.
# ---------------------------------------------------------------------------


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )


# ---------------------------------------------------------------------------
# cal_meanmass_vec (class_Mass_Bin.F90:1840-1849) + diag_pq's own icond1/em
# guard (mod_amps_core.F90:12692-12784, shared init before the phase
# branch), verbatim:
#
#   ms%mean_mass = ms%mass(1)/ms%con
#   if(ms%mean_mass==0.0_PS) em=10
#
#   ! diag_pq's own guard around the call:
#   if(con>1e-30 .and. mass(1)>1e-30) then
#     call cal_meanmass_vec(...)
#     if(em>0) then; icond1=1; con=0; mass(1)=0; mean_mass=0; endif
#   else
#     icond1=1; con=0; mass(1)=0; mean_mass=0
#   endif
# ---------------------------------------------------------------------------


def _mean_mass_and_active(con: np.ndarray, mass_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_mass, active) -- `active` is `icond1==0` (bins the
    Fortran actually computes physics for), the logical negation of the
    Fortran's own `icond1` flag."""
    valid = (con > 1.0e-30) & (mass_total > 1.0e-30)
    mean_mass = np.where(valid, _safe_div(mass_total, con), 0.0)
    active = valid & (mean_mass != 0.0)
    mean_mass = np.where(active, mean_mass, 0.0)
    return mean_mass, active


# ---------------------------------------------------------------------------
# cal_den_aclen_vec (class_Group.F90:11222-11300), verbatim.
# ---------------------------------------------------------------------------

_ALPHA_POLY_COEFFS = (1.001668, -0.098055, -2.52686, 3.75061, -1.68692)


def _alpha_poly(x: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3, c4 = _ALPHA_POLY_COEFFS
    return c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4


def _den_aclen(
    mean_mass: np.ndarray,
    con: np.ndarray,
    mass_aero_total: np.ndarray,
    eps_map: np.ndarray,
    *,
    den_as: float,
    den_ai: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (density, length, a_len, c_len, den_ap, r_n) -- `den_ap`/
    `r_n` are also needed by `_vapdep_coef` below, so returned rather than
    recomputed."""
    den_w = float(AmpsConst.den_w)
    pi = float(AmpsConst.PI)
    coef4pi3 = float(AmpsConst.coef4pi3)

    den_ap = _safe_div(np.full_like(eps_map, den_ai), 1.0 - eps_map * (1.0 - den_ai / den_as))

    map_ = _safe_div(mass_aero_total, con)  # per-particle aerosol mass
    density = _safe_div(mean_mass, _safe_div(mean_mass - map_, den_w) + _safe_div(map_, den_ap))

    r_n = _safe_div(map_, coef4pi3 * den_ap) ** (1.0 / 3.0)

    length = _safe_div(6.0 * mean_mass, pi * density) ** (1.0 / 3.0)
    length = np.maximum(r_n * 1.05, length)
    density = _safe_div(mean_mass, (pi / 6.0) * length**3)

    small = length <= 280.0e-4
    medium = (~small) & (length <= 1.0e-1)
    large = ~small & ~medium

    alpha_medium = np.maximum(_alpha_poly(length), 1.0e-5)
    dum_medium = length / (8.0 * alpha_medium) ** (1.0 / 3.0)

    dum_clamped = np.minimum(0.9, length)
    alpha_large = np.maximum(_alpha_poly(dum_clamped), 1.0e-5)
    dum_large = length / (8.0 * alpha_large) ** (1.0 / 3.0)

    a_len = np.select(
        [small, medium, large],
        [length / 2.0, dum_medium, dum_large],
    )
    c_len = np.select(
        [small, medium, large],
        [length / 2.0, dum_medium * alpha_medium, dum_large * alpha_large],
    )

    return density, length, a_len, c_len, den_ap, r_n


# ---------------------------------------------------------------------------
# cal_terminal_vel_vec, phase==1 branch (class_Group.F90:8069-8119),
# verbatim.
# ---------------------------------------------------------------------------

_VTM_MID_POLY = (
    -0.318657e1,
    0.992696,
    -0.153193e-2,
    -0.987059e-3,
    -0.578878e-3,
    0.855176e-4,
    -0.327815e-5,
)
_VTM_HIGH_POLY = (
    -0.500015e1,
    0.523778e1,
    -0.204914e1,
    0.475294,
    -0.542819e-1,
    0.238449e-2,
)


def _poly(coeffs: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for power, c in enumerate(coeffs):
        out = out + c * x**power
    return out


def _terminal_velocity(
    length: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    *,
    den_a: np.ndarray,
    d_vis: np.ndarray,
    sig_wa: np.ndarray,
) -> np.ndarray:
    gg = float(AmpsConst.gg)
    den_w = float(AmpsConst.den_w)
    rad = length * 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        # Branch B: Stokes, 0.5e-4 <= rad < 10e-4.
        u_s = rad**2 * gg * (den_w - den_a) / 4.5 / np.where(d_vis > 0.0, d_vis, 1.0)
        lambda_a = 6.6e-6 * (d_vis / 1.818e-4) * (1013250.0 / p) * (t / 293.15)
        vtm_stokes = (1.0 + 1.26 * lambda_a * _safe_div(np.ones_like(rad), rad)) * u_s

        # Branch C: 10e-4 <= rad < 535e-4.
        rad_c = np.where(rad > 0.0, rad, 1.0)
        x_c = np.log(32.0 * rad_c**3 * (den_w - den_a) * den_a * gg / 3.0 / d_vis**2)
        nre_c = np.exp(_poly(_VTM_MID_POLY, x_c))
        vtm_mid = nre_c * d_vis / (2.0 * rad_c * den_a)

        # Branch D: rad >= 535e-4 (rad clamped to 3500e-4 first).
        rad_d = np.where(rad > 0.0, np.minimum(rad, 3500.0e-4), 1.0)
        nbo = gg * (den_w - den_a) * rad_d**2 / sig_wa
        np_ = sig_wa**3 * den_a**2 / d_vis**4 / gg
        x_d = np.log(nbo * np_ ** (1.0 / 6.0) * 16.0 / 3.0)
        nre_d = np_ ** (1.0 / 6.0) * np.exp(_poly(_VTM_HIGH_POLY, x_d))
        vtm_high = nre_d * d_vis / (2.0 * rad_d * den_a)

        return np.select(
            [rad < 0.5e-4, rad < 10.0e-4, rad < 535.0e-4, rad >= 535.0e-4],
            [np.zeros_like(rad), vtm_stokes, vtm_mid, vtm_high],
        )


# ---------------------------------------------------------------------------
# cal_ventilation_coef_vec, phase==1 branch (G3 §3b), + get_fkn
# (class_Thermo_Var.F90:707-730), verbatim.
# ---------------------------------------------------------------------------


def _ventilation_piecewise(x: np.ndarray) -> np.ndarray:
    return np.select(
        [x < 1.4, (x >= 1.4) & (x <= 51.4), x > 51.4],
        [1.0 + 0.108 * x**2, 0.78 + 0.308 * x, np.full_like(x, 0.78 + 0.308 * 51.4)],
    )


def _ventilation(
    length: np.ndarray,
    vtm: np.ndarray,
    den: np.ndarray,
    *,
    d_vis: np.ndarray,
    d_v: np.ndarray,
    k_a: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_v = float(AmpsConst.R_v)
    pi = float(AmpsConst.PI)
    beta_w = 0.036
    delta = 1.0e-5

    n_sc = _safe_div(d_vis, den) / np.where(d_v > 0.0, d_v, 1.0)
    n_ns = _safe_div(d_vis, den) / np.where(k_a > 0.0, k_a, 1.0)
    nre = length * vtm * den / np.where(d_vis > 0.0, d_vis, 1.0)
    nre = np.maximum(nre, 0.0)

    x_v = n_sc ** (1.0 / 3.0) * np.sqrt(nre)
    x_h = n_ns ** (1.0 / 3.0) * np.sqrt(nre)
    fv = _ventilation_piecewise(x_v)
    fh = _ventilation_piecewise(x_h)

    r_m = length * 0.5
    r_m_safe = np.where(r_m > 0.0, r_m, 1.0)
    fkn_inv = r_m_safe / (r_m_safe + delta) + (d_v / beta_w / r_m_safe) * np.sqrt(
        2.0 * pi / (r_v * t)
    )
    fkn = 1.0 / fkn_inv

    return fv, fh, fkn


# ---------------------------------------------------------------------------
# cal_coef_vapdep2_vec, phase==1 branch (G3 §3c), verbatim.
# ---------------------------------------------------------------------------


def _vapdep_coef(
    cap: np.ndarray,
    fv: np.ndarray,
    a_len: np.ndarray,
    *,
    eps_map: np.ndarray,
    den_ap: np.ndarray,
    r_n: np.ndarray,
    d_v: np.ndarray,
    sig_wa: np.ndarray,
    e_sat1: np.ndarray,
    den: np.ndarray,
    t: np.ndarray,
    nu_aps0: float,
    m_aps0: float,
    phi_aps0: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_v = float(AmpsConst.R_v)
    pi = float(AmpsConst.PI)
    den_w = float(AmpsConst.den_w)
    m_w = float(AmpsConst.M_w)
    a_cliq = float(AmpsConst.a_cliq)
    l_e = float(AmpsConst.L_e)
    c_pa = float(AmpsConst.C_pa)

    r_n3 = r_n**3
    aa = 2.0 * sig_wa / (r_v * t * den_w)
    sb = nu_aps0 * eps_map * m_w * den_ap / (m_aps0 * den_w) * phi_aps0
    beta = 0.5
    s_salt = _safe_div(aa, a_len) - sb * r_n ** (2.0 * (1.0 + beta)) * _safe_div(
        np.ones_like(a_len), a_len**3 - r_n3
    )

    vw = np.sqrt(8.0 / pi * r_v * t)
    buzai_con = 4.0 * d_v / (vw * a_cliq)

    rho_s = e_sat1 / (t * r_v)
    gamma_w = 1.0 + l_e * rho_s / (c_pa * t * den) * (l_e / r_v / t - 1.0)

    coef1 = (
        4.0
        * pi
        * d_v
        * rho_s
        / gamma_w
        * cap
        * cap
        * _safe_div(np.ones_like(cap), cap + buzai_con)
        * fv
    )
    coef2 = -coef1 * s_salt
    return coef1, coef2


# ---------------------------------------------------------------------------
# diag_pq_liquid -- the phase==1 branch of diag_pq (G3 §5), orchestrating
# the helpers above.
# ---------------------------------------------------------------------------


def diag_pq_liquid(
    liquid_state: LiquidState, thermo: ThermoState, config: AmpsConfig, luts: AmpsLuts
) -> LiquidDiag:
    """`diag_pq`'s liquid (phase==1) branch, G3 §5 lines 12786-12835, plus
    the helper routines it calls (`cal_meanmass_vec`, `cal_den_aclen_vec`,
    `cal_terminal_vel_vec`, `cal_ventilation_coef_vec`,
    `cal_capacitance_vec`, `cal_coef_vapdep2_vec`) -- see the module
    docstring for exactly which G3 §5 outputs this exposes and the
    aerosol-category / dry-air-density notes.

    `luts` is accepted for interface parity with the task's specified
    signature but currently UNUSED: `cal_coef_vapdep2_vec`'s liquid branch
    uses the fixed molal coefficient `phi_aps` (a plain `AmpsConfig` scalar,
    not the osmotic-coefficient LUT `luts.osm_nh42so4`/`osm_sodchl` -- those
    LUTs are consumed by a different, more detailed CCN-activation routine,
    not by anything on this function's call path).

    Args:
        liquid_state: `LiquidState` (`ncat` must be 1, per `state.py`).
        thermo: `ThermoState` supplying P (`ptotv`), T (`tv`), moist-air
            density (`moist_denv`), and vapor mixing ratio (`qvv`).
        config: `AmpsConfig` (aerosol chemistry parameters).
        luts: `AmpsLuts` (unused here, see above).

    Returns:
        `LiquidDiag`, `(nbins, npoints)` per field.
    """
    del luts  # see docstring
    if liquid_state.ncat != 1:
        raise NotImplementedError(
            f"diag_pq_liquid only supports ncat=1 (see state.py's own to_fields()/"
            f"from_fields() convention); got ncat={liquid_state.ncat}"
        )

    con = liquid_state.values[LiquidPPV.rcon_q.py_idx, :, 0, :]
    mass_total = liquid_state.values[LiquidPPV.rmt_q.py_idx, :, 0, :]
    mass_aero_total = liquid_state.values[LiquidPPV.rmat_q.py_idx, :, 0, :]
    mass_aero_soluble = liquid_state.values[LiquidPPV.rmas_q.py_idx, :, 0, :]

    mean_mass, active = _mean_mass_and_active(con, mass_total)

    # diag_pq's own eps_map diagnosis (G3 §5, lines 1937-1938), only for
    # active (icond1==0) bins -- inactive bins keep the eps_ap0(ica) group
    # default (config.eps_ap[0], see module docstring); consistent with
    # den_as/den_ai below using the SAME category index.
    eps_map = np.where(
        active,
        np.clip(_safe_div(mass_aero_soluble, mass_aero_total), 0.0, 1.0),
        config.eps_ap[0],
    )

    # ThermoProp.ptotv is SI Pa (state.py's own UNIT CONTRACT note); both
    # thermo_fn.diffusivity (F1 SS3e docstring: "p is ambient pressure in
    # g/s^2/cm (CGS)") and _terminal_velocity below (its own `1013250.0`
    # CGS reference pressure, matching diffusivity's `p_0`) need CGS --
    # convert here, at the point of use.
    p = get_thermo_prop(thermo, ThermoProp.ptotv)[None, :] * 10.0
    t = get_thermo_prop(thermo, ThermoProp.tv)[None, :]
    den = get_thermo_prop(thermo, ThermoProp.moist_denv)[None, :]
    qv = get_thermo_prop(thermo, ThermoProp.qvv)[None, :]
    den_a = den * (1.0 - qv)  # see module docstring's dry-air-density note

    d_v = thermo_fn.diffusivity(p, t)
    k_a = thermo_fn.thermal_conductivity(t)
    d_vis = thermo_fn.dynamic_viscosity(t)
    sig_wa = thermo_fn.sfc_tension(t)
    estbar, esitbar = thermo_fn.make_esat_tables()
    e_sat1 = thermo_fn.esat_lk(1, np.broadcast_to(t, mean_mass.shape), estbar, esitbar)

    den_as0 = float(config.den_aps[0])
    den_ai0 = float(config.den_api[0])
    density, length, a_len, c_len, den_ap, r_n = _den_aclen(
        mean_mass, con, mass_aero_total, eps_map, den_as=den_as0, den_ai=den_ai0
    )

    den_a_b = np.broadcast_to(den_a, length.shape)
    d_vis_b = np.broadcast_to(d_vis, length.shape)
    sig_wa_b = np.broadcast_to(sig_wa, length.shape)
    p_b = np.broadcast_to(p, length.shape)
    t_b = np.broadcast_to(t, length.shape)
    den_b = np.broadcast_to(den, length.shape)
    d_v_b = np.broadcast_to(d_v, length.shape)
    k_a_b = np.broadcast_to(k_a, length.shape)

    vtm = _terminal_velocity(length, p_b, t_b, den_a=den_a_b, d_vis=d_vis_b, sig_wa=sig_wa_b)
    fv, fh, fkn = _ventilation(length, vtm, den_b, d_vis=d_vis_b, d_v=d_v_b, k_a=k_a_b, t=t_b)
    cap = 0.5 * length  # cal_capacitance_vec, phase==1: CAP = 0.5*len

    coef1, coef2 = _vapdep_coef(
        cap,
        fv,
        a_len,
        eps_map=eps_map,
        den_ap=den_ap,
        r_n=r_n,
        d_v=d_v_b,
        sig_wa=sig_wa_b,
        e_sat1=e_sat1,
        den=den_b,
        t=t_b,
        nu_aps0=float(config.nu_aps[0]),
        m_aps0=float(config.M_aps[0]),
        phi_aps0=float(config.phi_aps[0]),
    )

    def _select(computed: np.ndarray, default: float) -> np.ndarray:
        return np.where(active, computed, default)

    return LiquidDiag(
        mean_mass=mean_mass,
        length=_select(length, 0.0),
        a_len=_select(a_len, 0.0),
        c_len=_select(c_len, 0.0),
        density=_select(density, 1.0),
        terminal_velocity=_select(vtm, 0.0),
        capacitance=_select(cap, 0.0),
        ventilation_fv=_select(fv, 1.0),
        ventilation_fh=_select(fh, 1.0),
        ventilation_fkn=_select(fkn, 1.0),
        vapdep_coef1=_select(coef1, 0.0),
        vapdep_coef2=_select(coef2, 0.0),
    )

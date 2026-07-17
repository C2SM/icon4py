# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Saturation vapor pressure tables/accessors, thermodynamic coefficient
functions, and the theta_il closure, transcribed verbatim from AMPS Fortran
(scale_amps repo) per
docs/superpowers/facts/m1/constants-thermo.md ("F1" in docstrings below).

All numpy functions here are array-in/array-out, float64. The only DSL
deliverable is `_esat_lk_dsl` / `esat_lk_dsl` (liquid-phase table accessor,
TILED (Cell, K) table idiom -- see the section below for why).
"""

from __future__ import annotations

import gt4py.next as gtx
import numpy as np
import numpy.typing as npt
from gt4py.next import astype, maximum, minimum
from gt4py.next.experimental import as_offset

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


# ---------------------------------------------------------------------------
# Saturation vapor pressure table generation: QSPARM2 (F1 §3a, Murphy & Koop
# 2005), mod_amps_utility.F90:3012-3033.
# ---------------------------------------------------------------------------


def make_esat_tables() -> tuple[np.ndarray, np.ndarray]:
    """Build the Murphy-Koop (2005) saturation vapor pressure lookup tables.

    Transcribed verbatim from QSPARM2 (F1 §3a). T starts at 163. K and is
    incremented BEFORE each entry is evaluated, so Fortran entry K (1-based,
    K=1..150 for water / K=1..111 for ice) is evaluated at T = 163+K. In
    0-based numpy indexing, estbar[i] / esitbar[i] (i = 0..149 / i = 0..110)
    hold the value at T = 164+i.

    Tables are in Pa (see F1 §3a note); `esat_lk` multiplies by 10.0 to
    convert Pa -> dyn/cm^2 (CGS).

    Returns:
        (estbar, esitbar): estbar has 150 entries spanning T=164..313 K
        (liquid); esitbar has 111 entries spanning T=164..274 K (ice).
    """
    k_water = np.arange(1, 151, dtype=np.float64)
    t_water = 163.0 + k_water
    estbar = np.exp(
        54.842763
        - 6763.22 / t_water
        - 4.210 * np.log(t_water)
        + 0.000367 * t_water
        + np.tanh(0.0415 * (t_water - 218.8))
        * (53.878 - 1331.22 / t_water - 9.44523 * np.log(t_water) + 0.014025 * t_water)
    )

    k_ice = np.arange(1, 112, dtype=np.float64)
    t_ice = 163.0 + k_ice
    esitbar = np.exp(9.550426 - 5723.265 / t_ice + 3.53068 * np.log(t_ice) - 0.00728332 * t_ice)

    return estbar, esitbar


# ---------------------------------------------------------------------------
# Table-lookup accessor: get_sat_vapor_pres_lk (F1 §3c,
# class_Thermo_Var.F90:496-528).
# ---------------------------------------------------------------------------


def esat_lk(phase: int, t: npt.ArrayLike, estbar: np.ndarray, esitbar: np.ndarray) -> np.ndarray:
    """Table-lookup saturation vapor pressure, Fortran-exact semantics.

    phase: 1 = liquid (estbar, index cap 149), 2 = ice (esitbar, cap 110).

    Fortran (1-based I): I = MAX(1, MIN(INT(T)-163, cap));
    wt = MAX(MIN(T-real(I+163), 1.0), 0.0);
    e_sat = (estbar(I)*(1-wt) + estbar(I+1)*wt) * 10.0.

    INT(T) truncates toward zero; T is always positive in the physical
    range used here, so this is equivalent to floor(T). Truncation happens
    BEFORE the clamp to [1, cap] -- this order matters at the boundaries
    (see the truncate-vs-clamp regression test in test_thermo.py for a case
    where clamp-then-truncate would silently give a different, wrong
    result).

    1-based -> 0-based shift: Fortran estbar(I) is python estbar[I-1];
    Fortran estbar(I+1) is python estbar[I]. With idx = I-1 (0-based), the
    two gathered entries are estbar[idx] and estbar[idx+1].

    Returns e_sat in dyn/cm^2 (g/s^2/cm, CGS).
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if phase == 1:
        table, cap = estbar, 149
    elif phase == 2:
        table, cap = esitbar, 110
    else:
        raise ValueError(f"phase must be 1 (liquid) or 2 (ice), got {phase}")

    # Fortran INT(T): truncation toward zero (T > 0 throughout the physical
    # range used here, so np.trunc == floor).
    i_fortran = np.maximum(1, np.minimum(np.trunc(t_arr).astype(np.int64) - 163, cap))
    wt = np.clip(t_arr - (i_fortran + 163).astype(np.float64), 0.0, 1.0)

    idx = i_fortran - 1  # 0-based
    e0 = table[idx]
    e1 = table[idx + 1]
    return (e0 * (1.0 - wt) + e1 * wt) * 10.0


# ---------------------------------------------------------------------------
# Analytic saturation vapor pressure: get_sat_vapor_pres (F1 §3b, Lowe &
# Ficke 1974), class_Thermo_Var.F90:462-494.
# ---------------------------------------------------------------------------


def esat_analytic(phase: int, t: npt.ArrayLike) -> np.ndarray:
    """Analytic saturation vapor pressure, Lowe & Ficke (1974), verbatim.

    Returns e_sat in g/s^2/cm (CGS), including the x1000.0 conversion
    present in the Fortran source.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if phase == 1:
        e_sat = 6.1070 * np.exp(17.15 * (t_arr - 273.16) / (t_arr - 38.25))
    elif phase == 2:
        e_sat = 6.1064 * np.exp(21.88 * (t_arr - 273.16) / (t_arr - 7.65))
    else:
        raise ValueError(f"phase must be 1 (liquid) or 2 (ice), got {phase}")
    return 1000.0 * e_sat


# ---------------------------------------------------------------------------
# Reverse table lookup: get_T_fesv_lk (F1 §3d, class_Thermo_Var.F90:620-689).
# ---------------------------------------------------------------------------


def t_from_esat_lk(
    phase: int,
    t_guess: npt.ArrayLike,
    esat: npt.ArrayLike,
    estbar: np.ndarray,
    esitbar: np.ndarray,
) -> np.ndarray:
    """Reverse table lookup: T from e_sat, Fortran-exact (F1 §3d), verbatim.

    Search order: a +-5-entry window around the table index implied by
    `t_guess`, then (if no bracket found there) a full scan of entries
    below the window, then (if still none) a full scan of entries above the
    window -- exactly the three do-loops / goto-10 structure of
    get_T_fesv_lk, preserved as three passes with an early break.

    `esat` is divided by 10.0 before comparison against the table (the
    table is in Pa; esat_lk's caller-facing values are in CGS, dyn/cm^2 =
    Pa*10).

    Returns NaN where no table entry brackets `esat` in any of the three
    passes: the Fortran function leaves T_n undefined in that case (see F1
    §3d's note on `des`/T_n); NaN makes that failure explicit instead of
    silently returning uninitialized-memory semantics.
    """
    if phase == 1:
        table, cap = estbar, 149
    elif phase == 2:
        table, cap = esitbar, 110
    else:
        raise ValueError(f"phase must be 1 (liquid) or 2 (ice), got {phase}")

    t_guess_arr = np.asarray(t_guess, dtype=np.float64)
    esat_arr = np.asarray(esat, dtype=np.float64)
    shape = np.broadcast_shapes(t_guess_arr.shape, esat_arr.shape)
    flat_t = np.broadcast_to(t_guess_arr, shape).ravel()
    flat_es = np.broadcast_to(esat_arr / 10.0, shape).ravel()

    out_flat = np.full(flat_t.shape, np.nan, dtype=np.float64)

    for n in range(flat_t.size):
        t_n = flat_t[n]
        es = flat_es[n]
        i_guess = max(1, min(int(np.trunc(t_n)) - 163, cap))

        passes = (
            range(max(1, i_guess - 5), min(cap, i_guess + 5) + 1),
            range(1, i_guess - 4),
            range(i_guess + 5, cap + 1),
        )
        for window in passes:
            found = False
            for k in window:
                if table[k - 1] <= es < table[k]:
                    wt = (es - table[k - 1]) / (table[k] - table[k - 1])
                    out_flat[n] = (k + 163) + wt
                    found = True
                    break
            if found:
                break

    return out_flat.reshape(shape)


# ---------------------------------------------------------------------------
# Thermo coefficient functions (F1 §3e-3j, class_Thermo_Var.F90).
# ---------------------------------------------------------------------------


def diffusivity(p: npt.ArrayLike, t: npt.ArrayLike) -> np.ndarray:
    """D_v (cm^2/s): Hall & Pruppacher (1976), verbatim from
    get_diffusivity (F1 §3e, class_Thermo_Var.F90:381-392). p is ambient
    pressure in g/s^2/cm (CGS), t is temperature in K.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    p_0 = 1013250.0  # g/s^2/cm (function-local Fortran PARAMETER)
    return 0.211 * (t_arr / AmpsConst.T_0) ** 1.94 * (p_0 / p_arr)


def thermal_conductivity(t: npt.ArrayLike) -> np.ndarray:
    """k_a: Beard & Pruppacher (1971a), verbatim from
    get_thermal_conductivity (F1 §3f, class_Thermo_Var.F90:450-460).
    """
    t_arr = np.asarray(t, dtype=np.float64)
    return (5.69 + 0.017 * (t_arr - AmpsConst.T_0)) * 4.1868 * 1.0e2


def dynamic_viscosity(t: npt.ArrayLike) -> np.ndarray:
    """d_vis (g/cm/s): Pruppacher & Klett, verbatim from
    get_dynamic_viscosity (F1 §3g, class_Thermo_Var.F90:530-544).

    NOTE: uses the literal 273.15 (not AmpsConst.T_0 = 273.16, the triple
    point) -- transcribed verbatim; the Fortran source deliberately uses a
    different reference temperature here.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    tc = t_arr - 273.15
    mu_warm = (1.718 + 0.0049 * tc) * 1.0e-4
    mu_cold = (1.718 + 0.0049 * tc - 1.2e-5 * tc * tc) * 1.0e-4
    return np.where(tc >= 0.0, mu_warm, mu_cold)


def sfc_tension(t: npt.ArrayLike) -> np.ndarray:
    """sig_wa (erg/cm^2): Pruppacher & Klett (1997) eq. 5-12, verbatim from
    get_sfc_tension (F1 §3h, class_Thermo_Var.F90:545-559). Valid between
    -40 and 40 Celsius; Tc is clamped to [-45, 40] as in the Fortran
    source.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    an = (75.93, 0.115, 6.818e-2, 6.511e-3, 2.933e-4, 6.283e-6, 5.285e-8)
    tc = np.clip(t_arr - 273.15, -45.0, 40.0)
    return (
        an[0]
        + an[1] * tc
        + an[2] * tc * tc
        + an[3] * tc * tc * tc
        + an[4] * tc * tc * tc * tc
        + an[5] * tc * tc * tc * tc * tc
        + an[6] * tc * tc * tc * tc * tc * tc
    )


def gtp(  # noqa: PLR0917 [too-many-positional-arguments]
    t: npt.ArrayLike,
    p: npt.ArrayLike,
    d_v: npt.ArrayLike,
    k_a: npt.ArrayLike,
    es: npt.ArrayLike,
    iphase: int,
) -> np.ndarray:
    """Growth (diffusivity) function, verbatim from get_GTP (F1 §3i,
    class_Thermo_Var.F90:561-585). iphase: 1 = liquid (L_e), 2 = ice (L_s).

    `p` is accepted for interface parity with the Fortran signature
    (`d_P`) but is unused, matching the Fortran source (get_GTP declares
    but never reads d_P).
    """
    del p  # unused, matches Fortran get_GTP's unused d_P argument
    t_arr = np.asarray(t, dtype=np.float64)
    if iphase == 1:
        latent = AmpsConst.L_e
    elif iphase == 2:
        latent = AmpsConst.L_s
    else:
        raise ValueError(f"iphase must be 1 (liquid) or 2 (ice), got {iphase}")
    r_v = AmpsConst.R_v
    inv_gtp = r_v * t_arr / (es * d_v) + latent * (latent / (r_v * t_arr) - 1.0) / (k_a * t_arr)
    return 1.0 / inv_gtp


def mod_diffusivity(
    iphase: int, radius: npt.ArrayLike, od_v: npt.ArrayLike, t: npt.ArrayLike
) -> np.ndarray:
    """Modified (kinetically-corrected) diffusivity, verbatim from
    get_mod_diffusivity (F1 §3j, class_Thermo_Var.F90:393-429).

    NOTE: `a_cliq` here is the function-LOCAL Fortran PARAMETER = 1.0 (an
    alternate value 0.036 is commented out in the source), which is
    DISTINCT from AmpsConst.a_cliq = 0.036 (mod_amps_const.F90); transcribed
    verbatim, not substituted with the module constant.
    """
    radius_arr = np.asarray(radius, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    a_cliq_local = 1.0
    a_cice1 = 0.5
    del_v = 1.0e-5
    a_c = a_cliq_local if iphase == 1 else a_cice1
    denom = radius_arr / (radius_arr + del_v) + np.sqrt(
        2.0 * AmpsConst.PI / (AmpsConst.R_v * t_arr)
    ) * od_v / (radius_arr * a_c)
    return od_v / denom


def mod_thermal_cond(
    radius: npt.ArrayLike, ok_a: npt.ArrayLike, t: npt.ArrayLike, den: npt.ArrayLike
) -> np.ndarray:
    """Modified thermal conductivity, verbatim from get_mod_thermal_cond
    (F1 §3j, class_Thermo_Var.F90:431-446)."""
    radius_arr = np.asarray(radius, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    a_t = 0.96
    del_t = 2.16e-5
    denom = radius_arr / (radius_arr + del_t) + np.sqrt(
        2.0 * AmpsConst.PI * AmpsConst.M_a / (AmpsConst.R_u * t_arr)
    ) * ok_a / (den * radius_arr * a_t * AmpsConst.C_pa)
    return ok_a / denom


def mgtp(  # noqa: PLR0917 [too-many-positional-arguments]
    t: npt.ArrayLike,
    p: npt.ArrayLike,
    radius: npt.ArrayLike,
    d_v: npt.ArrayLike,
    k_a: npt.ArrayLike,
    den: npt.ArrayLike,
    es: npt.ArrayLike,
    iphase: int,
) -> np.ndarray:
    """Composite MGTP: modified diffusivity/conductivity fed through GTP,
    verbatim from update_DvKa (F1 §3j, class_Thermo_Var.F90:450-464)."""
    m_d_v = mod_diffusivity(iphase, radius, d_v, t)
    m_k_a = mod_thermal_cond(radius, k_a, t, den)
    return gtp(t, p, m_d_v, m_k_a, es, iphase)


# ---------------------------------------------------------------------------
# theta_il closure (F1 §5).
# ---------------------------------------------------------------------------


def diag_t(
    thil: npt.ArrayLike, p: npt.ArrayLike, qr: npt.ArrayLike, qi: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """T from theta_il + condensate loadings, verbatim from diag_t (F1 §5,
    mod_amps_core.F90:12449-12550), CGS constants (AmpsConst). qr, qi are
    hydrometeor mass mixing ratios (mass per unit moist-air mass; already
    divided by density, see F1 §5 notes).

    This is a direct closed-form branch selection, NOT an iterative
    fixed-point loop -- there is no loop in the quoted Fortran (a linear
    estimate assuming max(T,253)=253 is computed first; if that estimate is
    >=253, it is discarded in favor of the exact quadratic solution for the
    max(T,253)=T case).

    Returns (T, ierror1): ierror1[n] == 1 flags the Fortran-documented
    inconsistency where the quadratic branch's own result resolves to
    T < 253 K (the Fortran code keeps that result regardless; ierror1 is
    purely diagnostic, matching the source).
    """
    thil_arr = np.asarray(thil, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    qr_arr = np.asarray(qr, dtype=np.float64)
    qi_arr = np.asarray(qi, dtype=np.float64)

    til = thil_arr * (p_arr / AmpsConst.p00) ** AmpsConst.Racp
    heat = AmpsConst.L_e * qr_arr + AmpsConst.L_s * qi_arr

    t_lin = til * (1.0 + heat / (AmpsConst.C_pa * 253.0))
    t_quad = 0.5 * (til + np.sqrt(til * til + 4.0 * til / AmpsConst.C_pa * heat))

    needs_quad = t_lin >= 253.0
    t_out = np.where(needs_quad, t_quad, t_lin)
    ierror1 = np.where(needs_quad & (t_quad < 253.0), 1, 0)
    return t_out, ierror1


# F1 §5b's local Fortran PARAMETER values for cal_thetail (MKS). `p0` is
# deliberately absent from these defaults: F1's cal_thetail/cal_til
# reference a bare `p0` via implicit host-module association that is never
# declared/passed in the quoted subroutines (mod_amps_lib.F90:2217-2257) --
# a genuine fact-file gap (see task-2 report). cal_thetail/cal_til below
# therefore require `p0` as an explicit keyword argument rather than
# guessing its value; `cp`, `r`, `l_e`, `l_s` DO have F1-quoted verbatim
# defaults and are exposed as overridable keyword arguments purely so a
# caller can drive them with CGS-consistent equivalents (see
# test_cal_thetail_diag_t_round_trip_* in test_thermo.py).
CAL_THETAIL_L_E_MKS = 2.5e6  # J/kg
CAL_THETAIL_L_S_MKS = 2.8337e6  # J/kg
CAL_THETAIL_RA_MKS = 287.0  # J/K/kg
CAL_THETAIL_CP_MKS = 1004.0  # J/K/kg


def cal_thetail(
    qr: npt.ArrayLike,
    qi: npt.ArrayLike,
    pt: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    p0: float,
    cp: float = CAL_THETAIL_CP_MKS,
    r: float = CAL_THETAIL_RA_MKS,
    l_e: float = CAL_THETAIL_L_E_MKS,
    l_s: float = CAL_THETAIL_L_S_MKS,
) -> np.ndarray:
    """theta_il parcel-model closure, verbatim from cal_thetail (F1 §5b,
    mod_amps_lib.F90:2217-2257). See the `p0` note above CAL_THETAIL_L_E_MKS.
    """
    qr_arr = np.asarray(qr, dtype=np.float64)
    qi_arr = np.asarray(qi, dtype=np.float64)
    pt_arr = np.asarray(pt, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)

    pit = cp * (pt_arr / p0) ** (r / cp)
    theta = t_arr * cp / pit
    return theta / (1.0 + (l_e * qr_arr + l_s * qi_arr) / (cp * np.maximum(t_arr, 253.0)))


def cal_til(
    thetail: npt.ArrayLike,
    pt: npt.ArrayLike,
    *,
    p0: float,
    cp: float = CAL_THETAIL_CP_MKS,
    r: float = CAL_THETAIL_RA_MKS,
) -> np.ndarray:
    """theta_il -> T_il (Exner-function) conversion, verbatim from cal_til
    (F1 §5b, mod_amps_lib.F90:2217-2257). See the `p0` note above
    CAL_THETAIL_L_E_MKS.
    """
    thetail_arr = np.asarray(thetail, dtype=np.float64)
    pt_arr = np.asarray(pt, dtype=np.float64)
    pit = cp * (pt_arr / p0) ** (r / cp)
    return pit / cp * thetail_arr


# ---------------------------------------------------------------------------
# DSL deliverable: esat_lk (liquid, phase=1) via the TILED (Cell, K) table
# idiom -- the ONE DSL deliverable of this task.
#
# F5 §6 precedent (spike_a_remap_gather.py / spike_d_esat.py): a K-only
# table gathered via as_offset is NOT a portable idiom -- it is either a
# decoration-time DSLError (spike A) or backend-inconsistent (decorates but
# fails on embedded, "succeeds" on gtfn_cpu; spike D). The only idiom proven
# a GO on BOTH embedded and gtfn_cpu is the TILED (Cell, K) table: the table
# is memory-replicated across cells by the caller (`tile_estbar_for_dsl`
# below) so the gathered field already carries CellDim, matching the
# self-gather (`table(as_offset(Koff, ...))`) shape gt4py's dims deduction
# accepts.
#
# IMPORTANT: this reproduces esat_lk's Fortran-exact truncate-THEN-clamp
# order (I = MAX(1, MIN(INT(T)-163, 149)), then wt = clamp(T-(I+163), 0, 1)),
# NOT spike_d's clamp-then-truncate shortcut (`clip(T-163, 1, 149)` before
# truncating). The two orders coincide almost everywhere but diverge at the
# upper clamp boundary (e.g. T=350: Fortran selects the last table entry via
# wt=1; clamp-then-truncate would instead select the second-to-last entry
# via wt=0) -- see esat_lk's own docstring and test_thermo.py's
# truncate-vs-clamp regression case.
#
# `estbar_tiled` and `k_index` must each individually have >= 150 valid K
# positions (I ranges 1..149, 1-based, i.e. 0-based absolute gather targets
# 0..149) for the as_offset gather to be in-bounds; `t`/`out`'s own K-extent
# is independent of this and may be smaller (verified: gt4py permits
# t/table/k_index to have different K-extents in one field_operator call,
# each only needing enough entries for the positions it is actually
# accessed at -- see task-2 report for the probe).
# ---------------------------------------------------------------------------

ESAT_LK_DSL_TABLE_SIZE = 150  # estbar (liquid) length


def tile_estbar_for_dsl(estbar: np.ndarray, ncells: int) -> np.ndarray:
    """Memory-replicate the 1-D `estbar` table to (ncells, 150) for the
    TILED-table DSL idiom (see module docstring above)."""
    return np.tile(estbar, (ncells, 1))


@gtx.field_operator
def _esat_lk_dsl(
    t: fa.CellKField[ta.wpfloat],
    estbar_tiled: fa.CellKField[ta.wpfloat],
    k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
) -> fa.CellKField[ta.wpfloat]:
    """esat_lk(phase=1, ...) DSL equivalent -- liquid only. `estbar_tiled`
    is the 150-entry liquid table memory-replicated to (Cell, K) (see
    `tile_estbar_for_dsl`); `k_index` is arange(150) as a KDim field.
    """
    t_trunc = astype(t, gtx.int32)  # Fortran INT(T): truncation toward zero (T > 0)
    i_fortran = maximum(1, minimum(t_trunc - 163, 149))
    wt = maximum(minimum(t - astype(i_fortran + 163, ta.wpfloat), 1.0), 0.0)
    e0 = estbar_tiled(as_offset(Koff, i_fortran - 1 - k_index))
    e1 = estbar_tiled(as_offset(Koff, i_fortran - k_index))
    return (e0 * (1.0 - wt) + e1 * wt) * 10.0


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def esat_lk_dsl(
    t: fa.CellKField[ta.wpfloat],
    estbar_tiled: fa.CellKField[ta.wpfloat],
    k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
    out: fa.CellKField[ta.wpfloat],
) -> None:
    """Thin @gtx.program wrapper around `_esat_lk_dsl`."""
    _esat_lk_dsl(t, estbar_tiled, k_index, out=out)

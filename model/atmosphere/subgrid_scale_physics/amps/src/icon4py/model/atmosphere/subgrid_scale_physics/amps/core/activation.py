# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""AMPS CCN activation supersaturation solver -- CONTAINED PRIMITIVES only
(M2a Task 3, "Part 1"). The driver, `cal_aptact_var8_kc04dep` itself
(the outer grid-box loop, CCN critical-supersaturation precompute,
DHF/deposition Gauss-quadrature precompute, and activated-droplet
placement into bins), is a SEPARATE task ("Part 2" / Task 4) that will
import and call the functions in this module.

Ground truth: `docs/superpowers/facts/m2/activation.md` ("G2" below),
transcribing `cal_aptact_var8_kc04dep` (`mod_amps_core.F90:5776-9388`,
scale_amps repo) and its contained (Fortran-nested) routines. Per this
task's dispatch, G2's own "Contained solver routines (FULL bodies)"
section is authoritative and was transcribed VERBATIM for:
`cal_coef_svsteady_init`, `cal_air_temp`, `func_liqvap_vec`,
`func_icevap_vec`'s KC04-specific branches, `zbrent_act_vec`, `func_vec`.
Two helper functions G2 names but does not quote in full
(`get_critrad_anal`, `get_hazerad_anal`, both `class_Mass_Bin.F90`) were
read directly per the dispatch's explicit authorization ("reading the
named Fortran routine directly, quoting into report" for truncated
citations) -- quoted verbatim in each function's own docstring below.

Fortran "contained" (nested) subroutines close over a large amount of
HOST-SCOPE state (the outer `cal_aptact_var8_kc04dep`'s own local
variables/arrays, `ag`/`gr`/`gs`/`ga` derived-type members, and module-
level constants) via implicit Fortran host association -- Python has no
equivalent. Every such host variable a ported function needs is therefore
an EXPLICIT parameter here, grouped into small frozen dataclasses
(`LiquidBinState`, `IceBinState`, `ActivationBoxState`, `ActivationFlags`)
so call sites stay readable; each dataclass field's docstring cites the
exact Fortran variable it stands in for. Task 4 (the driver) is
responsible for computing/marshaling these arrays from the real
`LiquidState`/`IceState`/`AmpsConfig`/etc. -- this module intentionally
does NOT import `core.lookup_tables`/`config.AmpsConfig`/`core.index_maps`/
`state` (listed as "available for reuse" by the dispatch, not "must
import"): none of G2's contained-primitive bodies touch the osmotic-
coefficient LUTs or normal/inverse-normal CDF LUTs (those feed the DHF/
deposition Gauss-quadrature precompute in G2 section 1d/1e, which is
squarely driver/Task-4 territory), and the primitives operate on plain
`(nbins, npoints)`/`(npoints,)` numpy arrays rather than the `LiquidState`/
`IceState` PPV layout (a design choice explained below).

All arithmetic is CGS, float64. Solvers use MASKED FIXED-ITERATION numpy
(all lanes run the full iteration budget; a boolean "active" mask freezes
lanes that have already converged/failed) rather than reproducing
Fortran's dynamic index-compaction (`mbx`/`Lbx` shrinking every
iteration) -- mathematically equivalent (a frozen lane's inputs don't
change, so re-evaluating it is a no-op), and the idiom this task's binding
constraints explicitly call for.

Notable transcription/scope notes (see each function's own docstring for
the full citation):

1. `zbrent_act_vec`'s bracket-expansion fallback (`sw_n(n)=sw_o(n)` for
   `ierror1(n)>0` lanes, G2's own "bracketing failed -> fall back to old
   S" comment) is UNCONDITIONALLY OVERWRITTEN by the main Brent loop
   immediately afterward, for every lane -- verified against the Fortran
   itself: after the fallback block, `Lbx2` is reset to the FULL original
   `mbx` (not the bracket-failure-reduced subset), so every lane
   re-enters the main loop, and `sw_n(n)=b(n)` fires there at least once
   (either at the tolerance-converged check or the bisection/interpolation
   step) for every lane in `mbx`. This is confirmed vestigial/dead code in
   the ORIGINAL Fortran, not a porting bug -- ported here bit-for-bit
   anyway per the binding constraint ("same non-convergence fallback"),
   with `ierror1` still returned so a caller can independently observe
   which lanes had bracketing trouble even though it does not change
   `sw_n`'s final value.
2. `func_icevap_vec`'s base ice vapor-growth/melting physics (`bin_loop1`,
   G2 lines 8577-8914 -- narrated, not fully quoted, in G2's own "func_icevap_vec"
   section) was read directly from `mod_amps_core.F90` per the dispatch's
   authorization and IS fully ported, INCLUDING the 5-trial Lagrange-
   parabola ice-surface-temperature solve and the "whole ice melt" mass
   tendency -- EXCEPT the "partial melt + shed" sub-branch (triggered when
   `mean_mass - mass_ap > m_w`), which needs `cal_semiac_ip`/`get_vip`
   (`class_Ice_Shape.F90`) and per-bin ice-shape descriptors
   (`IS%den_ip`/`is_mod`/`phi_cs`). Confirmed by repo-wide grep: NO
   ice-shape port exists anywhere in this codebase (M1 or M2a); G5
   (`sedimentation-terminalvel.md`) itself only NAMES `get_vip` at a
   different call site, never quotes it -- so this is a genuine missing
   upstream prerequisite, not a truncated-citation gap this task's ground
   truth can resolve. Matching this codebase's own precedent
   (`core/lookup_tables.py`'s `make_breakup_fragment_tables`,
   `allow_placeholder=True` opt-in + `is_placeholder` result flag),
   `func_icevap_vec` raises `NotImplementedError` when the partial-melt
   -shed branch is actually reached UNLESS the caller passes
   `allow_shed_placeholder=True`, in which case `m_shed` is treated as 0
   (no shedding) and `IceVapResult.shed_is_placeholder` flags exactly
   which `(bin, point)` entries used the placeholder.
3. `ice_left`'s Fortran assignment is a plain OVERWRITE
   (`ice_left(n)=...`), not `+=`, unlike every other per-bin accumulator
   in `func_icevap_vec`/`func_liqvap_vec` (`used_Mi_vap`, `used_Mi_vapliq`,
   `loss_Mi_mlt`, `liq_left` are all genuine `x(n)=x(n)+...` accumulations
   -- confirmed by re-reading G2's own quoted text bin-by-bin). Because
   `bin_loop1` computes AND CONSUMES `ice_left(n)` for the SAME bin `j`
   within the SAME inner-loop iteration (before the next `j` overwrites
   it), this module computes melting-loss contributions from the PER-BIN,
   not point-summed, `ice_left` array, and reduces the function's
   RETURNED `ice_left` via an explicit "last valid bin wins" scan over
   the bin axis (in Fortran `j` order) -- NOT `.sum(axis=0)`. Getting this
   wrong would silently double-count evaporated/regrown ice mass across
   bins; flagged here because it is easy to miss on a first read.
4. `get_hazerad_anal` (`kohler_haze_radius`) declares its scratch
   variables `real`/`complex` (Fortran DEFAULT kind, i.e. single
   precision) rather than `real(PS)` (double) -- genuinely mixed
   precision in the ORIGINAL source (unlike `get_critrad_anal`, whose
   `Zd` is `real(PS)` and `Zdc`/`Pp`/`Pm` are explicit `real(8)`, i.e.
   double throughout). Per this task's CGS/float64 mandate, this port
   uses float64/complex128 uniformly; see `kohler_haze_radius`'s
   docstring.
5. `zbrent_act_vec` here takes a generic `residual_fn: Callable[[ndarray],
   ndarray]` rather than being hard-wired to `func_vec` -- this is a
   deliberate interface choice (not a Fortran feature: the Fortran
   `zbrent_act_vec` and `func_vec` are mutually the-only-caller/callee).
   It keeps the Brent-mechanics port (the binding constraint's main "port
   EXACTLY" target) independently testable against a synthetic monotone
   function, matching this task's own test spec, while a real caller
   (Task 4) supplies `residual_fn = lambda x: func_vec(x, t_a, iswitch,
   ...).residual`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import bin_grid, thermo
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import (
    LiquidMassIndex,
    LiquidPPV,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import AmpsLuts
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Module-level "local Fortran PARAMETER" constants, all sourced from G2 §0 /
# `zbrent_act_vec`'s own declarations (`mod_amps_core.F90:5960-5979,
# 9020-9024`) -- these are LOCAL constants of `cal_aptact_var8_kc04dep` /
# `zbrent_act_vec` themselves, not runtime `AmpsConfig` fields, hence
# hardcoded here rather than accepted as parameters (matching how
# `core/thermo.py`/`core/liquid_diag.py` hardcode their own Fortran
# PARAMETER literals).
# ---------------------------------------------------------------------------

#: `zbrent_act_vec`'s bracket-expansion iteration budget (`ITMAX_ini=30`,
#: distinct from the outer routine's own `ITMAX=200`).
ZBRENT_ITMAX_INI = 30
#: `zbrent_act_vec`'s main Brent-iteration budget (`ITMAX=50`).
ZBRENT_ITMAX = 50
#: Brent convergence tolerance (`tol=1.0e-6`).
ZBRENT_TOL = 1.0e-6
#: Brent machine-epsilon surrogate (`EPS=3.0e-8`).
ZBRENT_EPS = 3.0e-8
#: Water-saturation cap used by `func_vec`'s `iswitch==2` residual
#: (`sw_allow=0.02`, `mod_amps_core.F90:6124`).
SW_ALLOW = 0.02
#: Bin-validity number/mass thresholds shared by `func_liqvap_vec`/
#: `func_icevap_vec` (`nlmt=mlmt=1.0e-30`, `mod_amps_core.F90:5960`).
NLMT = 1.0e-30
MLMT = 1.0e-30

#: `func_icevap_vec`'s ice-surface-temperature Lagrange-parabola solve:
#: fixed half-window sizes for trials 1-5 (`dT_w=20.0,10.0,5.0,1.0,0.5`).
_TS_TRIAL_WINDOWS = (20.0, 10.0, 5.0, 1.0, 0.5)


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise `numerator/denominator`, 0.0 where `denominator<=0`,
    warnings suppressed -- same pattern as `core/liquid_diag.py`'s private
    helper of the same name (duplicated here per that module's own
    no-shared-utils precedent for small module-private numeric helpers)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )


def _esat_lk_by_phase(
    phase: npt.ArrayLike, t: npt.ArrayLike, estbar: np.ndarray, esitbar: np.ndarray
) -> np.ndarray:
    """`get_sat_vapor_pres_lk(phase2(j,n), t, ...)` where `phase2` is a
    per-(bin,point) array valued 1 (liquid) or 2 (ice) -- `core.thermo.
    esat_lk` only accepts a scalar `phase`, so this blends both table
    lookups via `np.where`. No `min(T_0, t)` clamp is applied here (some
    call sites need it, some don't -- callers clamp `t` themselves before
    calling, matching each individual Fortran call site)."""
    phase_arr = np.asarray(phase)
    e_liquid = thermo.esat_lk(1, t, estbar, esitbar)
    e_ice = thermo.esat_lk(2, t, estbar, esitbar)
    return np.where(phase_arr == 2, e_ice, e_liquid)


# ---------------------------------------------------------------------------
# Koehler critical/haze radius: get_critrad_anal / get_hazerad_anal
# (class_Mass_Bin.F90:2926-2954 / 3098-3158), read directly per this task's
# dispatch authorization (G2 names but does not quote these in full -- G2
# section "Koehler critical/haze radius usage" / section 3). Both are the
# ANALYTIC (Khvorostyanov & Curry 2014) variants preferred by the M0 spike;
# `get_critrad_itr`/`get_hazerad_itr` (Brent-iteration fallbacks, ALSO
# present in class_Mass_Bin.F90) are NOT ported -- G2 itself confirms they
# are "present but not used on the kc04dep production path" (section 3).
# ---------------------------------------------------------------------------


def kohler_critical_radius(
    aa: npt.ArrayLike, sb: npt.ArrayLike, beta: npt.ArrayLike, r_n: npt.ArrayLike
) -> np.ndarray:
    """Critical (activation) radius, `get_critrad_anal`
    (`class_Mass_Bin.F90:2926-2954`), Khvorostyanov & Curry (2014) eq
    6.3.7, verbatim:

        Zd=r_n**beta*sqrt(sb/(3.0_PS*AA))
        Zdc=Zd
        if(Zd>1.0e-2) then
          Pp=(Zdc**3+sqrt(Zdc**3+0.25_PS)+0.5_PS)**(1.0/3.0)
          Pm=(Zdc**3-sqrt(Zdc**3+0.25_PS)+0.5_PS)**(1.0/3.0)
          r_cr=r_n*(Zdc+Pp+Pm)
        else
          r_cr=r_n*(1.0d+0+Zdc+2.0d+0*Zdc**3/3.0d+0)
        endif

    All arguments/return in CGS (cm for `r_n`/return; `aa`/`sb` are the
    dimensionless Kelvin/solute coefficients the Fortran itself documents
    as "cm and dimensionless"). `aa` is the CALLER's `AA/T` (the Fortran
    call site always passes `AA/ag%TV(n)%T`, `AA=coef_a*sig_wa` with
    `coef_a=2.0/(den_w*R_v)` -- `mod_amps_core.F90:6314,6575` -- a 1-line
    driver/Task-4 computation, not reproduced here; a caller can build it
    as `2.0*core.thermo.sfc_tension(t)/(AmpsConst.den_w*AmpsConst.R_v*t)`,
    matching `core/liquid_diag.py`'s own `aa = 2.0*sig_wa/(r_v*t*den_w)`
    line for the identical Kelvin coefficient).

    `np.cbrt` (not `**(1.0/3.0)`) is used throughout so the real cube
    root of a negative argument (possible in the `Pm` branch) is handled
    correctly, matching Fortran's real `**(1.0/3.0)` semantics for a
    negative base (well-defined for a real cube root, unlike numpy's
    `**` operator, which returns NaN for a negative base with a
    fractional exponent).
    """
    aa_arr = np.asarray(aa, dtype=np.float64)
    sb_arr = np.asarray(sb, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    r_n_arr = np.asarray(r_n, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        zd = r_n_arr**beta_arr * np.sqrt(sb_arr / (3.0 * aa_arr))
    zd3 = zd**3
    inner = np.sqrt(zd3 + 0.25)
    pp = np.cbrt(zd3 + inner + 0.5)
    pm = np.cbrt(zd3 - inner + 0.5)

    r_cr_big = r_n_arr * (zd + pp + pm)
    r_cr_small = r_n_arr * (1.0 + zd + 2.0 * zd3 / 3.0)
    return np.where(zd > 1.0e-2, r_cr_big, r_cr_small)


def kohler_haze_radius(
    aa: npt.ArrayLike,
    sb: npt.ArrayLike,
    beta: npt.ArrayLike,
    s_w: npt.ArrayLike,
    r_n: npt.ArrayLike,
) -> np.ndarray:
    """Haze (equilibrium, sub-critical) radius, `get_hazerad_anal`
    (`class_Mass_Bin.F90:3098-3158`), Khvorostyanov & Curry (2014) eq
    6.2.12/6.2.28, verbatim, three regimes:

        Zd=r_n**beta*sqrt(sb/(3.0*AA))
        if(S_w<0.97.and.r_n>1.0e-6) then
          C_w=AA/(3.0*sb**(1.0/3.0)*r_n**(2.0*(1.0+beta)/3.0))
          r_w=r_n*(1.0+sb*r_n**(2.0*(1.0+beta)-3.0)/(-log(S_w))*
                (1.0+C_w*(-log(S_w))**(-2.0/3.0))**(-3.0))**(1.0/3.0)
        elseif(Zd<=0.1) then
          Zd2=Zd*Zd
          r_w=r_n*(1.0+Zd2-1.0/3.0*Zd2*Zd2*Zd2)
        else
          Zdc=cmplx(Zd,0.0)
          Aci=(1.0+sqrt(1.0-4.0*Zdc**6))**(1.0/3.0)
          Bci=(1.0-sqrt(1.0-4.0*Zdc**6))**(1.0/3.0)
          r_w=c2*r_n*(Aci+Bci)   ! c2=1/2**(1/3)=0.7937005259841
        endif

    NOTE (see module docstring item 4): the Fortran declares `Zd`/`Zd2` as
    default-kind `real` (single precision) and `Zdc`/`Aci`/`Bci` as
    default-kind `complex` (single-precision complex) here -- genuinely
    mixed precision relative to the rest of this (otherwise `real(PS)`
    =double) subroutine. This port computes uniformly in float64/
    complex128 per the task's CGS/float64 mandate; the resulting radius
    differs from a strict single-precision reproduction only at the
    ~1e-7 relative level in the third (complex-root) branch.

    Assigning a complex Fortran expression to the REAL `r_w` implicitly
    takes the real part (standard Fortran complex-to-real assignment
    semantics) -- reproduced here via `.real`.
    """
    aa_arr = np.asarray(aa, dtype=np.float64)
    sb_arr = np.asarray(sb, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    s_w_arr = np.asarray(s_w, dtype=np.float64)
    r_n_arr = np.asarray(r_n, dtype=np.float64)

    c2 = 0.7937005259841
    s_w_lmt = 0.97

    with np.errstate(divide="ignore", invalid="ignore"):
        zd = r_n_arr**beta_arr * np.sqrt(sb_arr / (3.0 * aa_arr))

        neg_log_sw = -np.log(s_w_arr)
        c_w = aa_arr / (3.0 * np.cbrt(sb_arr) * r_n_arr ** (2.0 * (1.0 + beta_arr) / 3.0))
        r_w_branch1 = r_n_arr * (
            1.0
            + sb_arr
            * r_n_arr ** (2.0 * (1.0 + beta_arr) - 3.0)
            / neg_log_sw
            * (1.0 + c_w * neg_log_sw ** (-2.0 / 3.0)) ** (-3.0)
        ) ** (1.0 / 3.0)

        zd2 = zd * zd
        r_w_branch2 = r_n_arr * (1.0 + zd2 - (1.0 / 3.0) * zd2 * zd2 * zd2)

        zdc = zd.astype(np.complex128)
        aci = (1.0 + np.sqrt(1.0 - 4.0 * zdc**6)) ** (1.0 / 3.0)
        bci = (1.0 - np.sqrt(1.0 - 4.0 * zdc**6)) ** (1.0 / 3.0)
        r_w_branch3 = (c2 * r_n_arr * (aci + bci)).real

    branch1_cond = (s_w_arr < s_w_lmt) & (r_n_arr > 1.0e-6)
    branch2_cond = zd <= 0.1
    return np.select(
        [branch1_cond, branch2_cond],
        [r_w_branch1, r_w_branch2],
        default=r_w_branch3,
    )


# ---------------------------------------------------------------------------
# cal_coef_svsteady_init (mod_amps_core.F90:8406-8414), verbatim (see the
# function's own docstring for the exact Fortran assignment it ports).
# ---------------------------------------------------------------------------


def cal_coef_svsteady_init(coef1_bin1: npt.ArrayLike, dt: float) -> np.ndarray:
    """Steady-state supersaturation-solve coefficient, `cal_coef_svsteady_init`
    (`mod_amps_core.F90:8406-8414`), verbatim: `a = coef(1)*dt`, where
    `coef(1)` is the FIRST liquid bin's (Fortran 1-based bin index 1)
    vapor-deposition growth coefficient (`core.liquid_diag.LiquidDiag.
    vapdep_coef1[0, :]`). `n` (the Fortran per-box index) is not a
    parameter here -- this is called once per grid box in the Fortran;
    this port is elementwise-vectorized over whatever shape
    `coef1_bin1` has (typically `(npoints,)`).
    """
    return np.asarray(coef1_bin1, dtype=np.float64) * dt


# ---------------------------------------------------------------------------
# cal_air_temp (mod_amps_core.F90:8416-8431), verbatim.
# ---------------------------------------------------------------------------


def cal_air_temp(til: npt.ArrayLike, qr: npt.ArrayLike, qi: npt.ArrayLike) -> np.ndarray:
    """Air temperature from `T_il` + condensate loadings, `cal_air_temp`
    (`mod_amps_core.F90:8416-8431`), verbatim:

        T=Til*(1.0_PS+(L_e*qr+L_s*qi)/(c_pa*253.0_PS))
        if(T>253.0_PS) then
           T=0.5*(Til+sqrt(Til**2+4.0_PS*Til/c_pa*(L_e*qr+L_s*qi)))
        endif

    NOTE: this is a DIFFERENT (simpler) function from `core.thermo.diag_t`
    (F1 §5): `diag_t` first applies the Exner conversion `til = thil*(P/
    p00)**Racp` and returns an extra `ierror1` diagnostic; `cal_air_temp`
    takes `Til` (already Exner-converted) directly and has no `ierror1`
    output, and its switch condition is strict `T>253.0` (not `diag_t`'s
    `t_lin>=253.0`) -- both transcribed exactly as their respective
    Fortran sources read, not unified.
    """
    til_arr = np.asarray(til, dtype=np.float64)
    qr_arr = np.asarray(qr, dtype=np.float64)
    qi_arr = np.asarray(qi, dtype=np.float64)

    l_e = float(AmpsConst.L_e)
    l_s = float(AmpsConst.L_s)
    c_pa = float(AmpsConst.C_pa)

    heat = l_e * qr_arr + l_s * qi_arr
    t_lin = til_arr * (1.0 + heat / (c_pa * 253.0))
    with np.errstate(invalid="ignore"):
        t_quad = 0.5 * (til_arr + np.sqrt(til_arr**2 + 4.0 * til_arr / c_pa * heat))
    return np.where(t_lin > 253.0, t_quad, t_lin)


# ---------------------------------------------------------------------------
# Per-bin / per-box state bundles -- see module docstring for why these
# exist (Fortran host-association has no Python equivalent). All array
# fields are `(nbins, npoints)` (bin state) or `(npoints,)` (box state)
# float64 unless noted; Task 4 is responsible for populating them from the
# real `LiquidState`/`IceState`/driver arrays.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LiquidBinState:
    """Per-liquid-bin arrays `func_liqvap_vec` needs, `(nbins, npoints)`.
    Field names/units mirror `gr%MS(j,n)%<field>` in G2's `func_liqvap_vec`."""

    con: np.ndarray  #: number concentration, `gr%MS(j,n)%con`
    mass_total: np.ndarray  #: total particle mass, `gr%MS(j,n)%mass(rmt)` (`LiquidMassIndex.rmt`)
    mass_aerosol: np.ndarray  #: aerosol mass component, `mass(rmat)` (`LiquidMassIndex.rmat`)
    coef1: np.ndarray  #: vapor-dep growth coeff 1, `gr%MS(j,n)%coef(1)` (`LiquidDiag.vapdep_coef1`)
    coef2: np.ndarray  #: vapor-dep growth coeff 2, `gr%MS(j,n)%coef(2)` (`LiquidDiag.vapdep_coef2`)
    r_act: np.ndarray  #: Koehler activation radius of this bin's own aerosol content, `%r_act`
    mean_mass: np.ndarray  #: mean particle mass, `gr%MS(j,n)%mean_mass`
    density: np.ndarray  #: bulk density, `gr%MS(j,n)%den`


@dataclasses.dataclass(frozen=True)
class IceBinState:
    """Per-ice-bin arrays `func_icevap_vec` needs, `(nbins, npoints)`.
    `ts_a1`/`ts_b11`/`ts_b12`/`ts_b13`/`ts_d1`/`tmax`/`phase2` are the
    `cal_coef_Ts3` surface-temperature-solve coefficients G2 section 1b
    names but does not quote (driver/Task-4 precompute, NOT ported by
    this module -- accepted here as given inputs, same status as
    `coef1`/`coef2`)."""

    con: np.ndarray  #: number concentration, `gs%MS(j,n)%con`
    mass_total: np.ndarray  #: total particle mass, `gs%MS(j,n)%mass(1)` (`imt`)
    mass_aerosol: np.ndarray  #: aerosol mass component, `mass(imat)`
    mass_meltwater: np.ndarray  #: melt-water mass component, `mass(imw)`
    mean_mass: np.ndarray  #: mean particle mass, `gs%MS(j,n)%mean_mass`
    coef1: np.ndarray  #: vapor-dep growth coeff 1, `gs%MS(j,n)%coef(1)`
    coef2: np.ndarray  #: vapor-dep growth coeff 2, `gs%MS(j,n)%coef(2)`
    phase2: np.ndarray  #: int, 1 (over-water) or 2 (over-ice) growth-phase selector, `phase2(j,n)`
    ts_a1: np.ndarray  #: `TS_A1(j,n)`
    ts_b11: np.ndarray  #: `TS_B11(j,n)`
    ts_b12: np.ndarray  #: `TS_B12(j,n)`
    ts_b13: np.ndarray  #: `TS_B13(j,n)`
    ts_d1: np.ndarray  #: `TS_D1(j,n)`
    tmax: np.ndarray  #: `Tmax(j,n)`, upper clamp for the surface-temp solve
    tmp_prev: np.ndarray  #: persistent per-bin surface-temp state, `gs%MS(j,n)%tmp`
    ldmassdt2: np.ndarray  #: riming/freeze-rate tendency, `gs%MS(j,n)%Ldmassdt(2)`
    e_l01: np.ndarray  #: ventilation/heat-exchange coefficient, `E_L01(j,n)`


@dataclasses.dataclass(frozen=True)
class ActivationBoxState:
    """Per-grid-box (`(npoints,)`) driver-computed state shared by
    `func_liqvap_vec`/`func_icevap_vec`/`func_vec` -- every field
    corresponds to a host-associated Fortran variable G2 documents as
    computed EARLIER in `cal_aptact_var8_kc04dep` (its own per-box init,
    section 1b/1c/1d/1e/1f, or the outer Backward-Euler loop, section 1h)
    and simply READ by the contained routines this module ports."""

    mes_rc: np.ndarray  #: int, mixed-phase presence code (1=none,2=liquid,3=ice,4=both)
    t: np.ndarray  #: actual (not trial) air temperature, `ag%TV(n)%T`
    til: np.ndarray  #: Exner-converted ice-liquid potential temperature, `Til(n)`
    qr_0: np.ndarray  #: liquid mixing ratio before this vapor-advance step, `qr_0(n)`
    qi_0: np.ndarray  #: ice mixing ratio before this vapor-advance step, `qi_0(n)`
    gain_mi_rim: np.ndarray  #: riming mass gain, `gain_Mi_rim(n)`
    gain_mi_frn: np.ndarray  #: freezing mass gain, `gain_Mi_frn(n)`
    den: np.ndarray  #: moist-air density, `ag%TV(n)%den`
    qtp: np.ndarray  #: total water mixing ratio, `qtp(n)`
    pressure: np.ndarray  #: ambient pressure, `ag%TV(n)%P`
    qr_b: np.ndarray  #: previous-iterate liquid mixing ratio, `qr_b(n)` (`func_vec` `iswitch==0`)
    qi_b: np.ndarray  #: previous-iterate ice mixing ratio, `qi_b(n)` (`func_vec` `iswitch==0`)
    rv: np.ndarray  #: saturation vapor mixing ratio, `ag%TV(n)%rv`
    sw: (
        np.ndarray
    )  #: box's baseline liquid supersaturation, `sw(n)` (`func_liqvap_vec`'s `x>0` gate)
    used_ma_act: np.ndarray  #: total activatable aerosol mass, `used_Ma_act(n)`
    akk_lmt: np.ndarray  #: CCN-number-cap activation-fraction limiter, `akk_lmt(n)`
    sw_allact: np.ndarray  #: "all activated" supersaturation offset, `sw_allact(n)`
    si_alldep: (
        np.ndarray
    )  #: "all activated" ice-supersaturation offset (deposition), `si_alldep(n)`
    used_ma_dep: np.ndarray  #: total depositable (dust) aerosol mass, `used_Ma_dep(n)`
    aerosol_dep_con: np.ndarray  #: dust (category 2) bin-1 number conc, `ga(2)%MS(1,n)%con`
    aerosol_dep_mass: np.ndarray  #: dust (category 2) bin-1 mass, `ga(2)%MS(1,n)%mass(1)`
    s_c_dhfmin: np.ndarray  #: minimum DHF critical supersaturation, `s_c_dhfmin(n)`
    ds_alldhf: np.ndarray  #: "all activated" DHF supersaturation span, `ds_allDHF(n)`
    akk_lmt_dhf: np.ndarray  #: DHF activation-fraction limiter, `akk_lmt_DHF(n)`
    used_ma_dhf: np.ndarray  #: total DHF-activatable aerosol mass, `used_Ma_DHF(n)`


@dataclasses.dataclass(frozen=True)
class ActivationFlags:
    """Scalar (not per-box) flags/parameters shared across a single
    `cal_aptact_var8_kc04dep` call."""

    flagp_r: int  #: liquid prediction flag, `flagp_r`
    flagp_s: int  #: solid-hydrometeor prediction flag, `flagp_s`
    iflg_inuc: int  #: ice-nucleation master switch, `iflg_inuc`
    iflg_dep: int  #: deposition-nucleation scheme selector (1=classical, else=Meyers), `iflg_dep`
    iflg_dhf: int  #: DHF (deliquescence-heterogeneous-freezing) switch, `iflg_dhf`
    level: int  #: complexity level, `level` (gates `mass_ap`/melt-tendency branches)
    dt: float  #: microphysics substep length, `gr%dt`/`gs%dt`


# ---------------------------------------------------------------------------
# Result bundles.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LiqVapResult:
    """`func_liqvap_vec`'s outputs, all `(npoints,)`."""

    used_mr_vap: np.ndarray
    used_mr_act: np.ndarray
    liq_left: np.ndarray
    noccnt: np.ndarray  #: int, 0 where any liquid activation occurred this call
    #: liquid activation fraction actually applied; 0.0 where `noccnt==1`.
    #: The Fortran leaves `akk(n)` at its prior host-scope value when the
    #: activation condition is false; this port has no such persistent
    #: state and returns 0.0 there instead -- inert, since `akk` is only
    #: ever meaningfully consumed elsewhere when `noccnt==0`.
    akk: np.ndarray


@dataclasses.dataclass(frozen=True)
class IceVapResult:
    """`func_icevap_vec`'s outputs, all `(npoints,)` except
    `shed_is_placeholder` (`(nbins, npoints)`, see module docstring item 2)."""

    used_mi_vap: np.ndarray
    used_mi_vapliq: np.ndarray
    used_mi_act: np.ndarray
    loss_mi_mlt: np.ndarray
    ice_left: np.ndarray  #: "last valid bin wins" -- see module docstring item 3
    noindep: np.ndarray  #: int, 0 where deposition-freezing occurred this call
    nodhft: np.ndarray  #: int, 0 where DHF occurred this call
    shed_is_placeholder: np.ndarray  #: bool `(nbins, npoints)`, see module docstring item 2


@dataclasses.dataclass(frozen=True)
class FuncVecResult:
    """`func_vec`'s outputs, all `(npoints,)`."""

    x_n: np.ndarray  #: self-consistently diagnosed liquid supersaturation
    y_n: np.ndarray  #: self-consistently diagnosed air temperature
    xi_n: np.ndarray  #: self-consistently diagnosed ice supersaturation
    residual: np.ndarray  #: the `iswitch`-dependent objective Brent drives to zero


@dataclasses.dataclass(frozen=True)
class ZbrentResult:
    sw_n: np.ndarray  #: converged (or best-effort, on non-convergence) root
    ierror1: (
        np.ndarray
    )  #: int, 0=ok, 1/2=bracket-expansion failed (a>1e4 / b<-2), 3=exhausted ITMAX_ini


# ---------------------------------------------------------------------------
# func_liqvap_vec (mod_amps_core.F90:8433-8530), verbatim.
# ---------------------------------------------------------------------------


def func_liqvap_vec(
    x: npt.ArrayLike,
    box: ActivationBoxState,
    liq: LiquidBinState,
    *,
    flagp_r: int,
    dt: float,
) -> LiqVapResult:
    """Liquid mass formed/left from activation + condensational growth,
    `func_liqvap_vec` (`mod_amps_core.F90:8433-8530`), verbatim.

    `x` is the trial liquid supersaturation (`(npoints,)`). Per bin, the
    three Fortran branches -- (1) full evaporation, (2) shrink below the
    bin's own Koehler-critical radius (return to haze), (3) normal
    growth/decay -- collapse to two DISTINCT actions here since branches
    (1) and (2) are IDENTICAL in the Fortran (`used_Mr_vap -= water_mass;
    liq_left += 0` in both): `evap_or_shrink = cond1 | cond2`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    npoints = x_arr.shape[0]

    used_mr_vap = np.zeros(npoints, dtype=np.float64)
    used_mr_act = np.zeros(npoints, dtype=np.float64)
    liq_left = np.zeros(npoints, dtype=np.float64)
    noccnt = np.ones(npoints, dtype=np.int64)
    akk = np.zeros(npoints, dtype=np.float64)

    if flagp_r <= 0:
        return LiqVapResult(used_mr_vap, used_mr_act, liq_left, noccnt, akk)

    with np.errstate(divide="ignore", invalid="ignore"):
        act_mask = (x_arr > 0.0) & (box.sw > 0.0) & (box.used_ma_act > 1.0e-25)
        akk_frac = np.where(
            box.sw_allact > 1.0e-25,
            np.minimum(
                1.0,
                np.minimum(
                    box.akk_lmt, x_arr / np.where(box.sw_allact > 1.0e-25, box.sw_allact, 1.0)
                ),
            ),
            np.minimum(1.0, np.maximum(0.0, box.akk_lmt)),
        )
        akk = np.where(act_mask, akk_frac, akk)
        used_mr_act = np.where(act_mask, akk * box.used_ma_act, used_mr_act)
        noccnt = np.where(act_mask, 0, noccnt)

        bin_valid = (
            ((box.mes_rc == 2) | (box.mes_rc == 4))[None, :]
            & (liq.con >= NLMT)
            & (liq.mass_total >= MLMT)
        )
        d_mean_mass = (liq.coef1 * x_arr[None, :] + liq.coef2) * dt
        water_mass = liq.mass_total - liq.mass_aerosol

        cond1 = d_mean_mass * liq.con <= (liq.mass_aerosol - liq.mass_total)
        shrink_radius = float(AmpsConst.coef3i4p1i3) * np.cbrt(
            _safe_div(liq.mean_mass + d_mean_mass, liq.density)
        )
        cond2 = (d_mean_mass < 0.0) & (liq.r_act > shrink_radius)
        evap_or_shrink = cond1 | cond2

        d_used_mr_vap = np.where(evap_or_shrink, -water_mass, d_mean_mass * liq.con)
        d_liq_left = np.where(evap_or_shrink, 0.0, water_mass + d_mean_mass * liq.con)

        d_used_mr_vap = np.where(bin_valid, d_used_mr_vap, 0.0)
        d_liq_left = np.where(bin_valid, d_liq_left, 0.0)

        used_mr_vap = used_mr_vap + d_used_mr_vap.sum(axis=0)
        liq_left = liq_left + d_liq_left.sum(axis=0)

    return LiqVapResult(used_mr_vap, used_mr_act, liq_left, noccnt, akk)


# ---------------------------------------------------------------------------
# func_icevap_vec (mod_amps_core.F90:8532-8990), verbatim (see module
# docstring items 2/3 for the two documented scope decisions).
# ---------------------------------------------------------------------------


def func_icevap_vec(  # noqa: PLR0915, PLR0912, PLR0917 -- single verbatim Fortran subroutine
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    box: ActivationBoxState,
    ice: IceBinState,
    flags: ActivationFlags,
    estbar: np.ndarray,
    esitbar: np.ndarray,
    *,
    allow_shed_placeholder: bool = False,
) -> IceVapResult:
    """Ice mass, melting, deposition-freezing + DHF, `func_icevap_vec`
    (`mod_amps_core.F90:8532-8990`), verbatim -- see module docstring
    items 2 (partial-melt-shed placeholder) and 3 (`ice_left`
    overwrite-not-accumulate semantics) for the two load-bearing
    transcription notes.

    `x`: trial liquid supersaturation, `y`: reference air temperature for
    this evaluation (both `(npoints,)`, matching `func_icevap_vec(...,x,y)`'s
    own dummy arguments).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    npoints = x_arr.shape[0]
    nbins = ice.con.shape[0]

    used_mi_vap = np.zeros(npoints, dtype=np.float64)
    used_mi_vapliq = np.zeros(npoints, dtype=np.float64)
    used_mi_act = np.zeros(npoints, dtype=np.float64)
    loss_mi_mlt = np.zeros(npoints, dtype=np.float64)
    ice_left = np.zeros(npoints, dtype=np.float64)
    noindep = np.ones(npoints, dtype=np.int64)
    nodhft = np.ones(npoints, dtype=np.int64)
    shed_is_placeholder = np.zeros((nbins, npoints), dtype=bool)

    if flags.flagp_s <= 0:
        return IceVapResult(
            used_mi_vap,
            used_mi_vapliq,
            used_mi_act,
            loss_mi_mlt,
            ice_left,
            noindep,
            nodhft,
            shed_is_placeholder,
        )

    t_0 = float(AmpsConst.T_0)
    l_s = float(AmpsConst.L_s)
    l_e = float(AmpsConst.L_e)
    l_f = float(AmpsConst.L_f)
    c_w = float(AmpsConst.c_w)

    with np.errstate(all="ignore"):
        e_satw = thermo.esat_lk(1, y_arr, estbar, esitbar)
        e_sati = thermo.esat_lk(2, np.minimum(t_0, y_arr), estbar, esitbar)
        r_e = e_satw / e_sati
        xi = r_e * (x_arr + 1.0) - 1.0
        xw = x_arr

        bin_valid = (
            ((box.mes_rc == 3) | (box.mes_rc == 4))[None, :]
            & (ice.con >= NLMT)
            & (ice.mass_total >= MLMT)
        )

        # ---- 5-trial Lagrange-parabola ice surface-temperature solve ----
        ts_gate = (t_0 > box.t) & (t_0 > y_arr)  # (npoints,)
        ts_gate_bc = np.broadcast_to(ts_gate[None, :], (nbins, npoints))

        ts_b1 = np.where(
            ice.phase2 == 1,
            ice.ts_b11 * e_satw[None, :] / y_arr[None, :] * (xw[None, :] + 1.0)
            + ice.ts_b12
            + ice.ts_b13 * y_arr[None, :],
            ice.ts_b11 * e_sati[None, :] / y_arr[None, :] * (xi[None, :] + 1.0)
            + ice.ts_b12
            + ice.ts_b13 * y_arr[None, :],
        )

        tmp = ice.tmp_prev.copy()
        gate_ok = np.ones((nbins, npoints), dtype=bool)
        for trial_idx, dt_w in enumerate(_TS_TRIAL_WINDOWS):
            run_trial = ts_gate_bc if trial_idx < 2 else (ts_gate_bc & gate_ok)
            if not run_trial.any():
                if trial_idx >= 2:
                    break
                continue

            tmp_bf = tmp
            x2 = np.minimum(ice.tmax, tmp_bf + dt_w)
            x1 = x2 - dt_w
            x0 = x1 - dt_w
            gx0 = _esat_lk_by_phase(ice.phase2, x0, estbar, esitbar) / x0
            gx1 = _esat_lk_by_phase(ice.phase2, x1, estbar, esitbar) / x1
            gx2 = _esat_lk_by_phase(ice.phase2, x2, estbar, esitbar) / x2
            w0 = gx0 / ((x0 - x1) * (x0 - x2))
            w1 = gx1 / ((x1 - x0) * (x1 - x2))
            w2 = gx2 / ((x2 - x0) * (x2 - x1))
            a_l = w0 + w1 + w2
            b_l = -w0 * (x1 + x2) - w1 * (x0 + x2) - w2 * (x0 + x1)
            d_l = w0 * x1 * x2 + w1 * x0 * x2 + w2 * x0 * x1

            ts_d2 = (ice.ts_d1 * b_l + ice.ts_a1) ** 2 - 4.0 * ice.ts_d1 * a_l * (
                ice.ts_d1 * d_l - ts_b1
            )
            sqrt_d2 = np.sqrt(np.maximum(ts_d2, 0.0))
            tmp_candidate = np.where(
                ts_d2 >= 0.0,
                (-(ice.ts_d1 * b_l + ice.ts_a1) + sqrt_d2) / (2.0 * ice.ts_d1 * a_l),
                tmp_bf,
            )
            tmp_candidate = np.minimum(tmp_candidate, ice.tmax)

            new_tmp = np.where(run_trial, tmp_candidate, tmp)
            if trial_idx >= 1:
                diff_ratio = np.abs(new_tmp - tmp_bf) / tmp_bf
                gate_ok = np.where(run_trial, diff_ratio > 1.0e-4, gate_ok)
            tmp = new_tmp

        tmp = np.where(ts_gate_bc, tmp, t_0)

        # ---- depositional growth / evaporation ----
        esat_tmp = _esat_lk_by_phase(ice.phase2, tmp, estbar, esitbar)
        d_mean_mass_ice = (
            ice.coef1 * e_sati[None, :] / y_arr[None, :] * (xi[None, :] + 1.0)
            + ice.coef2 * esat_tmp / tmp
        ) * flags.dt
        d_mean_mass_liq = (
            ice.coef1 * e_satw[None, :] / y_arr[None, :] * (xw[None, :] + 1.0)
            + ice.coef2 * esat_tmp / tmp
        ) * flags.dt
        is_ice_branch = ice.phase2 == 2
        d_mean_mass = np.where(is_ice_branch, d_mean_mass_ice, d_mean_mass_liq)

        water_mass = ice.mass_total - ice.mass_aerosol
        evaporated = _safe_div(water_mass, ice.con) + d_mean_mass <= 0.0

        d_used = np.where(evaporated, -water_mass, d_mean_mass * ice.con)
        # NOTE: overwrite semantics, NOT accumulated -- see module
        # docstring item 3. `d_ice_left` stays per-bin (unsummed).
        d_ice_left = np.where(evaporated, 0.0, water_mass + d_mean_mass * ice.con)

        used_mi_vap = np.where(bin_valid & is_ice_branch, d_used, 0.0).sum(axis=0)
        used_mi_vapliq = np.where(bin_valid & ~is_ice_branch, d_used, 0.0).sum(axis=0)

        ice_left = np.zeros(npoints, dtype=np.float64)
        for j in range(nbins):
            ice_left = np.where(bin_valid[j], d_ice_left[j], ice_left)

        # ---- melting mass m_w ----
        m_w_ice_melt = (
            (
                l_s
                * (
                    ice.coef1 * e_sati[None, :] / y_arr[None, :] * (xi[None, :] + 1.0)
                    + ice.coef2 * esat_tmp / tmp
                )
                + (l_f + c_w * (y_arr[None, :] - tmp)) * ice.ldmassdt2
                - ice.e_l01 * (tmp - y_arr[None, :])
            )
            * flags.dt
            / l_f
        )
        m_w_ice = np.where(tmp == t_0, m_w_ice_melt, 0.0)

        t_bc = np.broadcast_to(box.t[None, :], (nbins, npoints))
        m_w_liq_warm = np.maximum(
            0.0,
            (
                l_e
                * (
                    ice.coef1 * e_satw[None, :] / y_arr[None, :] * (x_arr[None, :] + 1.0)
                    + ice.coef2 * esat_tmp / tmp
                )
                + c_w * (y_arr[None, :] - tmp) * ice.ldmassdt2
                - ice.e_l01 * (tmp - y_arr[None, :])
            )
            * flags.dt
            / l_f,
        )
        m_w_liq_cold_melt = (
            (
                l_e
                * (
                    ice.coef1 * e_satw[None, :] / y_arr[None, :] * (x_arr[None, :] + 1.0)
                    + ice.coef2 * esat_tmp / tmp
                )
                + (l_f + c_w * (y_arr[None, :] - tmp)) * ice.ldmassdt2
                - ice.e_l01 * (tmp - y_arr[None, :])
            )
            * flags.dt
            / l_f
        )
        m_w_liq_cold = np.where(tmp == t_0, m_w_liq_cold_melt, 0.0)
        m_w_liq = np.where(t_0 <= t_bc, m_w_liq_warm, m_w_liq_cold)

        m_w = np.where(is_ice_branch, m_w_ice, m_w_liq)

        # ---- melting-tendency accounting ----
        if flags.level <= 3:
            mass_ap = np.zeros_like(ice.mass_aerosol)
        else:
            mass_ap = _safe_div(ice.mass_aerosol, ice.con)
        dm_w = m_w
        meltwater_per_particle = _safe_div(ice.mass_meltwater, ice.con)
        m_w_total = meltwater_per_particle + m_w

        if flags.level <= 5:
            melt_gate = m_w_total > 0.0
        else:
            melt_gate = (m_w_total > 0.0) & ((meltwater_per_particle > 1.0e-12) | (dm_w != 0.0))
        melt_gate = melt_gate & bin_valid

        whole_melt = melt_gate & (ice.mean_mass - mass_ap <= m_w_total)
        partial_melt_shed = melt_gate & ~whole_melt

        if partial_melt_shed.any() and not allow_shed_placeholder:
            raise NotImplementedError(
                "func_icevap_vec: partial-melt-with-shedding branch requires "
                "cal_semiac_ip/get_vip (class_Ice_Shape.F90) and per-bin ice-shape "
                "descriptors (IS%den_ip/is_mod/phi_cs) -- not ported anywhere in "
                "this codebase (confirmed: no ice-shape port exists in M1/M2a; "
                "see core/activation.py's module docstring item 2). Pass "
                "allow_shed_placeholder=True to treat m_shed=0 (no shedding) for "
                "the affected bins -- IceVapResult.shed_is_placeholder flags "
                "exactly which (bin, point) entries used the placeholder."
            )

        loss_whole = np.where(whole_melt, np.maximum(0.0, np.minimum(d_ice_left, water_mass)), 0.0)
        shed_is_placeholder = partial_melt_shed
        loss_mi_mlt = loss_whole.sum(axis=0)  # partial_melt_shed contributes 0.0 (placeholder)

        # ---- KC04 deposition-freezing branch ----
        if flags.iflg_inuc > 0 and flags.iflg_dep > 0:
            # DELIBERATE DIVERGENCE from the literal Fortran, documented per
            # this project's divergence policy (port the INTENT, document
            # the divergence, flag upstream -- do not replicate an
            # accidental aliasing bug). Fortran (`mod_amps_core.F90:
            # 8919-8925`) computes `flg1=x(n)` OUTSIDE (before) the
            # `do m=1,Lbx` box loop it is later tested inside, using
            # whatever `n` happens to be left over from a PRECEDING loop in
            # the same subroutine -- i.e. every box's classical-nucleation
            # (`iflg_dep==1`) gate is keyed off ONE arbitrary, stale box
            # index's supersaturation, not its own. That reads as an
            # accidental Fortran scalar-aliasing artifact (a leftover loop
            # variable reused post-loop), not intended physics -- there is
            # no plausible physical reason box A's deposition gate should
            # depend on box B's `x`. This port instead evaluates `flg1`
            # per-box (`x_arr`, the array of EACH box's own trial
            # supersaturation), which is what the surrounding code's own
            # intent (a per-box classical-vs-Meyers selector) clearly
            # requires. Flagged upstream:
            # docs/superpowers/facts/m1/fortran-mapping.md "known
            # divergences" section (scale_amps repo).
            flg1 = x_arr if flags.iflg_dep == 1 else np.full(npoints, -1.0)
            dep_mask = (
                (xi > 0.0)
                & (y_arr < t_0)
                & (flg1 < 0.0)
                & (box.aerosol_dep_con >= NLMT)
                & (box.aerosol_dep_mass >= MLMT)
            )
            akk_dep = np.where(
                box.si_alldep > 1.0e-25,
                np.minimum(1.0, xi / np.where(box.si_alldep > 1.0e-25, box.si_alldep, 1.0)),
                0.0,
            )
            used_mi_act = np.where(dep_mask, used_mi_act + akk_dep * box.used_ma_dep, used_mi_act)
            noindep = np.where(dep_mask, 0, noindep)

        # ---- KC04 DHF (deliquescence-heterogeneous-freezing) branch ----
        if flags.iflg_inuc > 0 and flags.iflg_dhf > 0:
            dhf_gate = x_arr > box.s_c_dhfmin
            ratio = _safe_div(x_arr - box.s_c_dhfmin, box.ds_alldhf)
            akk_dhf = np.where(
                box.ds_alldhf > 1.0e-25,
                np.clip(np.minimum(box.akk_lmt_dhf, ratio), 0.0, 1.0),
                0.0,
            )
            dhf_apply = dhf_gate & (akk_dhf > 1.0e-6)
            used_mi_act = np.where(dhf_apply, used_mi_act + akk_dhf * box.used_ma_dhf, used_mi_act)
            nodhft = np.where(dhf_apply, 0, nodhft)

    return IceVapResult(
        used_mi_vap,
        used_mi_vapliq,
        used_mi_act,
        loss_mi_mlt,
        ice_left,
        noindep,
        nodhft,
        shed_is_placeholder,
    )


# ---------------------------------------------------------------------------
# func_vec (mod_amps_core.F90:9277-9387), verbatim.
# ---------------------------------------------------------------------------


def func_vec(  # noqa: PLR0917 [too-many-positional-arguments]
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    iswitch: int,
    box: ActivationBoxState,
    liq: LiquidBinState,
    ice: IceBinState,
    flags: ActivationFlags,
    estbar: np.ndarray,
    esitbar: np.ndarray,
    *,
    allow_shed_placeholder: bool = False,
) -> FuncVecResult:
    """The top-level residual `zbrent_act_vec` drives to zero, `func_vec`
    (`mod_amps_core.F90:9277-9387`), verbatim.

    `iswitch` selects the residual (Fortran `select case`):

    * `1`: fixed-point self-consistency residual -- condensate diagnosed
      at the trial `x` vs. condensate re-diagnosed at the resulting
      `x_n` (`zbrent_act_vec`'s stage-1 call, "vapor-advance" root).
    * `2`: water-saturation-cap residual, `-x_n + SW_ALLOW`
      (`zbrent_act_vec`'s stage-2 call, "water-saturation adjustment").
    * `0`: oscillation-check residual against `box.qr_b`/`box.qi_b` (no
      Brent solve -- the single direct `func_vec` call G2 section 1i
      describes right after `zbrent_act_vec`'s stage-1 call).

    The Fortran's own `em(n)` (a diagnostic side-output, `iswitch+10` or
    `0`, never read elsewhere in the quoted source) is intentionally not
    returned.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    liqvap = func_liqvap_vec(x_arr, box, liq, flagp_r=flags.flagp_r, dt=flags.dt)
    icevap = func_icevap_vec(
        x_arr,
        y_arr,
        box,
        ice,
        flags,
        estbar,
        esitbar,
        allow_shed_placeholder=allow_shed_placeholder,
    )

    with np.errstate(all="ignore"):
        trans_mi = icevap.loss_mi_mlt - np.minimum(
            liqvap.liq_left, box.gain_mi_rim + box.gain_mi_frn
        )
        qr = np.maximum(
            0.0, box.qr_0 + (liqvap.used_mr_act + liqvap.used_mr_vap + trans_mi) / box.den
        )
        qi = np.maximum(
            0.0,
            box.qi_0
            + (icevap.used_mi_vap + icevap.used_mi_vapliq + icevap.used_mi_act - trans_mi)
            / box.den,
        )

        y_n = cal_air_temp(box.til, qr, qi)
        e_satw = thermo.esat_lk(1, y_n, estbar, esitbar)
        qv_n = np.maximum(0.0, box.qtp - qr - qi)
        x_n = box.pressure * qv_n / (float(AmpsConst.Rdvchiarui) + qv_n) / e_satw - 1.0
        e_sati = thermo.esat_lk(2, np.minimum(float(AmpsConst.T_0), y_n), estbar, esitbar)
        r_e = e_satw / e_sati
        xi_n = r_e * (x_n + 1.0) - 1.0

        if iswitch == 1:
            liqvap2 = func_liqvap_vec(x_n, box, liq, flagp_r=flags.flagp_r, dt=flags.dt)
            icevap2 = func_icevap_vec(
                x_n,
                y_n,
                box,
                ice,
                flags,
                estbar,
                esitbar,
                allow_shed_placeholder=allow_shed_placeholder,
            )
            trans_mi2 = icevap2.loss_mi_mlt - np.minimum(
                liqvap2.liq_left, box.gain_mi_rim + box.gain_mi_frn
            )
            qr2 = np.maximum(
                0.0, box.qr_0 + (liqvap2.used_mr_act + liqvap2.used_mr_vap + trans_mi2) / box.den
            )
            qi2 = np.maximum(
                0.0,
                box.qi_0
                + (icevap2.used_mi_vap + icevap2.used_mi_vapliq + icevap2.used_mi_act - trans_mi2)
                / box.den,
            )
            residual = (qr + qi - qr2 - qi2) / np.maximum(1.0e-9, box.rv)
        elif iswitch == 2:
            residual = -x_n + SW_ALLOW
        elif iswitch == 0:
            residual = (qr + qi - box.qr_b - box.qi_b) / np.maximum(1.0e-9, box.rv)
        else:
            raise ValueError(f"iswitch must be 0, 1, or 2; got {iswitch}")

    return FuncVecResult(x_n=x_n, y_n=y_n, xi_n=xi_n, residual=residual)


# ---------------------------------------------------------------------------
# zbrent_act_vec (mod_amps_core.F90:8992-9275), verbatim -- see module
# docstring items 1 and 5.
# ---------------------------------------------------------------------------


def zbrent_act_vec(  # noqa: PLR0915, PLR0917 -- single verbatim Fortran subroutine
    residual_fn: Callable[[np.ndarray], np.ndarray],
    iphase: npt.ArrayLike,
    iswitch: int,
    sw_o: npt.ArrayLike,
    t_a_o: npt.ArrayLike,
    estbar: np.ndarray,
    esitbar: np.ndarray,
) -> ZbrentResult:
    """Masked-fixed-iteration Brent root-finder, `zbrent_act_vec`
    (`mod_amps_core.F90:8992-9275`), verbatim: same `ITMAX_ini=30`/
    `ITMAX=50` iteration budgets, same bracket-expansion
    doubling/additive-step + failure bounds (`a>1.0e4`/`b<-2.0`), same
    `tol=1.0e-6`/`EPS=3.0e-8` tolerance, same non-convergence fallback
    (see module docstring item 1 for why it is a provable no-op, ported
    anyway).

    `residual_fn(x) -> ndarray`: the objective to root-find (see module
    docstring item 5 for why this is a generic callable rather than a
    hard-wired `func_vec` call -- a real caller passes
    `lambda x: func_vec(x, t_a, iswitch, ...).residual`; this task's own
    test suite passes a synthetic monotone function directly).

    `iphase`: int array, 1 (pure-liquid bracket) or 2 (ice-referenced
    bracket, needs `t_a_o`/`estbar`/`esitbar` to convert the `-0.5`
    ice-supersaturation seed to an equivalent liquid supersaturation).
    G2 only documents these two values (the Fortran `if/elseif` has no
    `else`); any other `iphase` value here silently collapses to the
    `iphase==1` bracket (`b=-0.5`), not an error -- callers must only
    ever pass 1 or 2.
    `iswitch`: passed through unchanged to `residual_fn` calls in the
    ORIGINAL Fortran (via `func_vec`'s own `iswitch` argument) -- here it
    only controls the bracket-expansion step's doubling-vs-additive
    choice for `a` (`iswitch==2` multiplies, else adds `0.2`), matching
    `zbrent_act_vec`'s own (Fortran) use of `iswitch` (it does NOT forward
    `iswitch` into `residual_fn` -- the caller's closure already captures
    it, if relevant).

    `sw_b`/`T_a` (declared `intent(in)` in the Fortran signature) are
    OMITTED here: `sw_b` is never referenced anywhere in the quoted body
    (a genuinely unused Fortran dummy argument); `T_a` is exactly what a
    `residual_fn` closure captures instead of needing as a separate
    zbrent-level parameter.
    """
    iphase_arr = np.asarray(iphase)
    sw_o_arr = np.asarray(sw_o, dtype=np.float64)
    t_a_o_arr = np.asarray(t_a_o, dtype=np.float64)
    npoints = sw_o_arr.shape[0]

    # ---- initial brackets [b, a] per phase (9028-9042) ----
    a = np.full(npoints, 0.2, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_satw_o = thermo.esat_lk(1, t_a_o_arr, estbar, esitbar)
        e_sati_o = thermo.esat_lk(2, np.minimum(float(AmpsConst.T_0), t_a_o_arr), estbar, esitbar)
        r_e_o = e_satw_o / e_sati_o
        b_iphase2 = (-0.5 + 1.0) / r_e_o - 1.0
    b = np.where(iphase_arr == 2, b_iphase2, -0.5)

    ierror1 = np.zeros(npoints, dtype=np.int64)

    # ---- bracket-expansion phase (ITMAX_ini=30) ----
    active = np.ones(npoints, dtype=bool)
    fa = residual_fn(a)
    fb = residual_fn(b)
    for _ in range(ZBRENT_ITMAX_INI):
        if not active.any():
            break

        both_neg = (fa < 0.0) & (fb < 0.0)
        both_pos = (fa > 0.0) & (fb > 0.0)

        upd_a = active & both_neg
        a = np.where(upd_a, a * 2.0, a) if iswitch == 2 else np.where(upd_a, a + 0.2, a)
        fail_a = upd_a & (a > 1.0e4)

        upd_b = active & both_pos
        b = np.where(upd_b, b - 0.2, b)
        fail_b = upd_b & (b < -2.0)

        ierror1 = np.where(fail_a, 1, ierror1)
        ierror1 = np.where(fail_b, 2, ierror1)

        bracketed = active & ~both_neg & ~both_pos
        active = active & ~(bracketed | fail_a | fail_b)

        if not active.any():
            break
        # IMPORTANT: gate the refresh by the (just-updated) `active` mask,
        # not unconditional. A lane that goes inactive THIS iteration
        # (bracketed, or just failed via fail_a/fail_b) must have its
        # fa/fb FROZEN at the pre-update values (matching Fortran's
        # instant removal from the compacted `mbx2` -- that lane never
        # receives another `call func_vec`, so its fa(n)/fb(n) stay at
        # whatever they were computed as BEFORE the failing/bracketing
        # update). Recomputing unconditionally here would instead evaluate
        # `residual_fn` at the JUST-updated (possibly out-of-bounds, e.g.
        # `b<-2.0`) a/b for that lane -- a real divergence from Fortran
        # when a fast-failing/fast-bracketing lane is batched with a
        # still-active (slower) lane (a lane-by-lane loop has no such
        # issue; only the vectorized/batched case can expose it). See
        # `TestZbrentActVec.test_fast_failing_lane_matches_single_lane_result`.
        fa = np.where(active, residual_fn(a), fa)
        fb = np.where(active, residual_fn(b), fb)

    ierror1 = np.where(active, 3, ierror1)

    # "Non-convergence fallback" (Fortran: sw_n(n)=sw_o(n) for
    # ierror1(n)>0): see module docstring item 1 -- unconditionally
    # overwritten by the main Brent loop below for every lane, kept for
    # transcription fidelity; `ierror1` is still returned.
    sw_n = sw_o_arr.copy()

    # ---- main Brent iteration (ITMAX=50) ----
    active = np.ones(npoints, dtype=bool)
    c = b.copy()
    fc = fb.copy()
    d = np.zeros(npoints, dtype=np.float64)
    e = np.zeros(npoints, dtype=np.float64)

    for _ in range(ZBRENT_ITMAX):
        if not active.any():
            break
        with np.errstate(all="ignore"):
            same_sign = ((fb > 0.0) & (fc > 0.0)) | ((fb < 0.0) & (fc < 0.0))
            m1 = active & same_sign
            c = np.where(m1, a, c)
            fc = np.where(m1, fa, fc)
            d = np.where(m1, b - a, d)
            e = np.where(m1, b - a, e)

            # NOTE: Fortran executes these SEQUENTIALLY (a(n)=b(n); b(n)=c(n);
            # c(n)=a(n); ...), so "c(n)=a(n)" reads the JUST-UPDATED a(n)
            # (=old b), not the original a -- i.e. this is NOT a clean 3-cycle
            # rotation (new_c does NOT end up as old_a; it ends up equal to
            # new_a=old_b). Verified against the classic Numerical-Recipes-F77
            # ZBRENT source this subroutine is adapted from (identical text at
            # mod_amps_core.F90:9186-9193) -- reproduced here exactly via the
            # same read-after-write ordering, not "corrected" to a symmetric
            # rotation.
            m2 = active & (np.abs(fc) < np.abs(fb))
            a_new = np.where(m2, b, a)
            b_new = np.where(m2, c, b)
            c_new = np.where(m2, a_new, c)
            fa_new = np.where(m2, fb, fa)
            fb_new = np.where(m2, fc, fb)
            fc_new = np.where(m2, fa_new, fc)
            a, b, c = a_new, b_new, c_new
            fa, fb, fc = fa_new, fb_new, fc_new

            tol1 = 2.0 * ZBRENT_EPS * np.abs(b) + 0.5 * ZBRENT_TOL
            xm = 0.5 * (c - b)

            converged = active & ((np.abs(xm) <= tol1) | (fb == 0.0))
            sw_n = np.where(converged, b, sw_n)
            still = active & ~converged

            use_interp = still & (np.abs(e) >= tol1) & (np.abs(fa) > np.abs(fb))
            s = np.where(fa != 0.0, fb / fa, 0.0)
            a_eq_c = a == c
            qf = np.where(fc != 0.0, fa / fc, 0.0)
            rf = np.where(fc != 0.0, fb / fc, 0.0)
            p = np.where(
                a_eq_c,
                2.0 * xm * s,
                s * (2.0 * xm * qf * (qf - rf) - (b - a) * (rf - 1.0)),
            )
            q = np.where(a_eq_c, 1.0 - s, (qf - 1.0) * (rf - 1.0) * (s - 1.0))
            q = np.where(p > 0.0, -q, q)
            p = np.abs(p)

            accept = use_interp & (
                2.0 * p < np.minimum(3.0 * xm * q - np.abs(tol1 * q), np.abs(e * q))
            )

            d_interp = np.where(q != 0.0, p / q, xm)
            d_new = np.where(still, np.where(accept, d_interp, xm), d)
            e_new = np.where(still, np.where(accept, d, xm), e)
            d, e = d_new, e_new

            a = np.where(still, b, a)
            fa = np.where(still, fb, fa)

            step_big = still & (np.abs(d) > tol1)
            b = np.where(step_big, b + d, np.where(still, b + np.copysign(tol1, xm), b))
            sw_n = np.where(still, b, sw_n)

        active = still
        if not active.any():
            break
        # Same defensive gating as the bracket-expansion phase above (see
        # its comment). Provably a no-op HERE specifically: `b` is only
        # ever updated where `still` is True (the assignment two lines
        # above), so a lane that just went inactive (converged) this
        # iteration has an unchanged `b`, and `residual_fn(b)` would
        # recompute the identical value anyway -- gated regardless, for
        # symmetry with the bracket-expansion fix and to stay robust
        # against a future edit that adds another inactivation path here.
        fb = np.where(active, residual_fn(b), fb)

    return ZbrentResult(sw_n=sw_n, ierror1=ierror1)


# =============================================================================
# PART 2 (M2a Task 4): cal_aptact_var8_kc04dep -- the OUTER DRIVER.
#
# Ground truth: G2 (docs/superpowers/facts/m2/activation.md), sections
# 1a-1n ("OUTER STRUCTURE") + section 3 (`add_simple_vec`, quoted FULL).
# Several G2 sections are NARRATED, not fully quoted (1c's mass-bin-
# boundary construction, 1d/1e's Gauss-quadrature DHF/deposition
# precompute, 1f's `sw_allact`/`ds_allDHF`/`si_alldep` formula, 1k's
# post-`add_simple_vec` `cal_lincubprms_vec`/`cal_transbin_vec` machinery)
# -- per this task's dispatch ("dispatch authorizes reading the named
# Fortran directly, quoting into report"), these were read directly from
# `mod_amps_core.F90` (scale_amps repo) and `class_Mass_Bin.F90`; each
# resulting divergence/simplification is documented at its own point below
# AND summarized here:
#
# 1. CCN mass-bin quadrature (G2 section 1c's `zn`/`fact_c`/
#    `interp_data1d_lut_big(snrml,...)` machinery,
#    `mod_amps_core.F90:6564-6644`, read directly): this numerically
#    re-partitions ONE bulk lognormal aerosol population per category
#    (`ga(ica)%MS(1,n)` -- ALWAYS Fortran bin index 1; the Fortran's own
#    aerosol Group carries only a SINGLE lognormal-summary bin per
#    category, not a discretized population -- `AerosolState` bin index 0
#    stands in for it here, its own bin axis beyond index 0 UNUSED by this
#    driver) into `N_bin_a=10` pieces via a fixed 10-point Gauss-Hermite
#    quadrature (`ZN_QUADRATURE`/`FACT_C_QUADRATURE`, G2 section 0,
#    verbatim) driven by the category's own lognormal shape parameters
#    `p(1)` (geometric mean radius)/`p(2)` (ln-sigma). `p(1)`/`p(2)` are
#    THEMSELVES diagnosed (read directly: `diag_pardis_ap`,
#    `class_Mass_Bin.F90:1743-1789`, named nowhere in G2) from the
#    category's prognostic `mean_mass`/`den` (from `AerosolState` bin 0's
#    own `amt_q`/`acon_q`) plus a per-category ln-sigma constant
#    `ap_lnsig` -- initially believed to be an unported namelist parameter
#    (the ONLY per-category "mean/sigma" fields this module's own author
#    first found, `ap_mean_cp`/`ap_sig_cp`, are used EXCLUSIVELY by G2
#    section 1d's DHF contact-angle Gauss-quadrature,
#    `mod_amps_core.F90:6836,6961`; `ap_sig_cp`'s cloudlab values, 20.0
#    for 3 of 4 categories, are implausible as a lognormal geometric-sigma
#    but match a Savre & Ekman 2014 contact-angle sigma in degrees
#    exactly) -- but `config.py` DOES carry the correct fields under
#    different names: `config.ap_lnsig`/`config.ap_mean`
#    ("log of standard deviation (natural log of radius, microns)" /
#    "geometrical mean radius (cm)", both already M1-ported with real
#    per-category cloudlab values). `interp_data1d_lut_big` itself (named,
#    not quoted, by G2) was resolved from a commented-out INLINE
#    equivalent immediately above its own call site in the Fortran (see
#    `_interp_data1d_lut_big`'s docstring) -- so this section is now FULLY
#    ported, not simplified; the one remaining, explicitly documented
#    ASSUMPTION is WHICH of `diag_pardis_ap`'s branches applies
#    (`flagp` IN `{1,2,-1,-2,-3}`, "fixed standard deviation" -- see
#    `_ccn_precompute`'s own docstring for the justification), since G2
#    itself never names `diag_pardis_ap` at all (read purely to resolve
#    this driver's own precompute, out of G2's own quoted scope).
#
# 2. `sw_allact`/`ds_allDHF`/`si_alldep` (G2 section 1f, narrated as "the
#    'all-activated' saturation offsets" but not quoted): read directly
#    (`mod_amps_core.F90:7055-7079`) and ported verbatim below
#    (`_all_activated_supersaturation`).
#
# 3. Activated-droplet bin placement (G2 section 1k): the literal Fortran
#    calls `add_simple_vec` (G2 section 3, quoted FULL) ONLY for the "mean
#    mass falls outside its own shifted per-particle bin" edge case
#    (`icond4==1`, `mod_amps_core.F90:7620-7663`); the common case goes
#    through `cal_lincubprms_vec`/`cal_transbin_vec`
#    (`mod_amps_core.F90:7682,7745`), NEITHER quoted by G2 NOR ported
#    anywhere in this codebase -- read directly, confirmed genuinely
#    un-quotable within this task's scope (a general sub-bin linear/cubic
#    mass-redistribution engine reused by most AMPS bin processes, not
#    specific to activation, per its own module comment "calculation of
#    parameters for linear distribution in the shifted bin"). Per this
#    project's missing-prerequisite precedent (Task 3's ice-shape shed
#    branch), this port uses `add_simple_vec` (fully quoted, genuine
#    Fortran physics) for ALL activated-droplet placement, not just the
#    narrow edge case -- correctly conserving number and mass
#    (`add_simple_vec`'s own placement algorithm does this by
#    construction) at the cost of not reproducing the sub-bin linear
#    redistribution the literal Fortran uses for the common case.
#    Documented prominently; see the task report.
#
# 4. Deliquescence-heterogeneous-freezing (`DHF_IF1`/`DHF_IF2`, G2 section
#    4) and the classical-CNT deposition-nucleation precompute (`Dep_IF1`,
#    G2 section 1e's `iflg_dep==1` branch): both are full Savre & Ekman
#    (2014)/classical-nucleation-theory 40-point Gauss-quadrature
#    integrals G2 narrates (names the intermediate quantities: haze
#    radius, molality, `sigma_iv/sigma_sv/sigma_is`, `Lmef`, `Gn`, `Hvfr`,
#    `r_cr`, `dFact`, `r_d`) but does not quote the formulas for -- a
#    second genuine missing-prerequisite, comparable in scope to Task 3's
#    ice-shape gap. `iflg_dhf`/`iflg_dep` gate these branches independent
#    of `flagp_s` in the literal Fortran, but BOTH are physically
#    freezing/deposition processes that deposit mass into ICE bins -- this
#    port's actual integration point (`implementations/warm_loop.py`'s
#    warm-phase-only `_activation`) has no ice group at all, and
#    cloudlab's own config defaults DHF off (`ice_nucleation_dhf=0`) and
#    gates deposition on a genuine dust (category-2) aerosol population
#    this port's warm-only wiring never supplies. Matching Task 3's
#    `allow_shed_placeholder` precedent: the DHF/deposition precompute
#    below raises `NotImplementedError` when its flag is genuinely on AND
#    applicable (DHF: `iflg_dhf>0`; deposition: `iflg_dep>0 AND
#    aerosol.ncat>=2`, i.e. a dust category is actually present) UNLESS
#    `allow_dhf_placeholder=True`/`allow_dep_placeholder=True`, in which
#    case it degrades to a provably-inert no-op (matching the Fortran's
#    own pre-loop-init values, `s_c_dhfmin=999.0`,
#    `used_Ma_dhf=used_Ma_dep=0`) -- never silently skipped.
# =============================================================================

MIN_EPS_CCN = 1.0e-6  # min soluble mass fraction for CCN eligibility (mod_amps_core.F90:36)
DRIVER_MAX_R_N = 10.0  # micron; max dry radius for CCN eligibility (G2 section 0)
MAX_FACT = 1.0  # clamp on the CCN number/mass integration fraction (G2 section 0)
OUTER_ITMAX = 200  # backward-Euler Til-Qt loop iteration budget (G2 section 0)
ALIM1 = 1.0e-5  # backward-Euler vapor-mixing-ratio convergence tolerance (G2 section 0)
ALIM2 = 1.0e-5  # backward-Euler temperature convergence tolerance (G2 section 0)
S_C_DHFMIN_INIT = 999.0  # Fortran pre-loop init value, mod_amps_core.F90:6466

#: `N_bin_a`, the number of on-the-fly CCN mass-quadrature bins (G2 section
#: 0/1c) -- fixed at 10 by the `zn`/`fact_c` node counts below, not a
#: runtime `AmpsConfig` field.
N_BIN_A = 10
#: 10-point Gauss-Hermite quadrature BOUNDARIES in standardized
#: log-radius z-score space (`zn`, `mod_amps_core.F90:6307-6309`, verbatim,
#: 11 values bounding `N_BIN_A=10` bins).
ZN_QUADRATURE = np.array(
    [
        -3.090232306,
        -2.472185845,
        -1.854139384,
        -1.236092922,
        -0.618046461,
        0.0,
        0.618046461,
        1.236092922,
        1.854139384,
        2.472185845,
        3.090232306,
    ]
)
#: per-bin NUMBER-fraction weights (`fact_c`, `mod_amps_core.F90:6310-6311`,
#: verbatim; sums to ~1.0).
FACT_C_QUADRATURE = np.array(
    [
        0.005714484,
        0.025145127,
        0.076352402,
        0.160060344,
        0.231727644,
        0.231727644,
        0.160060344,
        0.076352402,
        0.025145127,
        0.005714484,
    ]
)


def _interp_data1d_lut_big(lut, x: np.ndarray) -> np.ndarray:
    """`interp_data1d_lut_big`, a linear-interpolation lookup into a
    `Data1DLut` (`y[i]` sampled at `xs+dx*i`). G2 does not quote this
    function's body (named, not shown); its call site
    (`mod_amps_core.F90:~6598-6612`) has a commented-out INLINE equivalent
    immediately above the actual call, read directly and transcribed
    verbatim here (0-based, `i1_fortran-1`):

        i1=max(1,min(n-1,int((x-xs)/dx)+1))   ! 1-based, clamped to [1,n-1]
        x1=real(i1-1)*dx+xs
        wx=min(1,max(0,(x-x1)/dx))
        y_out=min(1,max(0,(1-wx)*y(i1)+wx*y(i1+1)))
    """
    i1 = np.clip(np.floor((x - lut.xs) / lut.dx).astype(np.int64), 0, lut.n - 2)
    x1 = i1 * lut.dx + lut.xs
    wx = np.clip((x - x1) / lut.dx, 0.0, 1.0)
    return np.clip((1.0 - wx) * lut.y[i1] + wx * lut.y[i1 + 1], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Small thermo helpers standing in for Fortran `AirGroup` diagnostic fields
# (`ag%TV(n)%rv_sat`, `%s_v`) this port's simpler `ThermoState` does not
# carry persistently -- all derived from first principles (P/T/qv/esat),
# using the SAME formula `func_vec`'s own diagnosis uses (G2's own
# `x_n`/`xi_n` assignments), not guessed.
# ---------------------------------------------------------------------------


def _rv_saturation(
    p: np.ndarray, t: np.ndarray, estbar: np.ndarray, esitbar: np.ndarray
) -> np.ndarray:
    """Saturation vapor mixing ratio, standing in for `ag%TV(n)%rv_sat(1)`:
    the `sw=0` inverse of `sw=P*qv/(Rdvchiarui+qv)/e_satw-1`, i.e.
    `rv_sat=Rdvchiarui*e_satw/(P-e_satw)`."""
    e_satw = thermo.esat_lk(1, t, estbar, esitbar)
    rdv = float(AmpsConst.Rdvchiarui)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rdv * e_satw / np.where(p > e_satw, p - e_satw, 1.0)


def _liquid_supersaturation(
    p: np.ndarray, qv: np.ndarray, t: np.ndarray, estbar: np.ndarray, esitbar: np.ndarray
) -> np.ndarray:
    """`sw=P*qv/(Rdvchiarui+qv)/e_satw(T)-1`, standing in for the initial
    `ag%TV(n)%s_v(1)` -- same formula as `func_vec`'s own `x_n`."""
    e_satw = thermo.esat_lk(1, t, estbar, esitbar)
    rdv = float(AmpsConst.Rdvchiarui)
    return p * qv / (rdv + qv) / e_satw - 1.0


def _ice_supersaturation(
    sw: np.ndarray, t: np.ndarray, estbar: np.ndarray, esitbar: np.ndarray
) -> np.ndarray:
    """`si=r_e*(sw+1)-1`, `r_e=e_satw(T)/e_sati(min(T_0,T))` -- standing in
    for the initial `ag%TV(n)%s_v(2)`, same formula as `func_vec`'s
    `xi_n`."""
    t_0 = float(AmpsConst.T_0)
    e_satw = thermo.esat_lk(1, t, estbar, esitbar)
    e_sati = thermo.esat_lk(2, np.minimum(t_0, t), estbar, esitbar)
    return e_satw / e_sati * (sw + 1.0) - 1.0


def _invert_cal_air_temp(t: np.ndarray, qr: np.ndarray, qi: np.ndarray) -> np.ndarray:
    """Closed-form inverse of `cal_air_temp` (Part 1) for `til`:
    `activate_and_advance_vapor`'s own dispatched signature is given
    (T, qr, qi) via `ThermoState`/derived `qr_0`/`qi_0`, not `til` itself
    (unlike `WarmLoopState`, which carries `thil` as a SEPARATE per-column
    scalar the Fortran's outer `cal_micro_tendency` threads through --
    out of this function's own signature). `cal_air_temp`'s two branches
    invert in closed form: linear (`T=Til*(1+heat/(c_pa*253))` ->
    `Til=T/(1+heat/(c_pa*253))`) and quadratic
    (`T=0.5*(Til+sqrt(Til^2+4*Til/c_pa*heat))` -> `Til=T^2/(T+heat/c_pa)`,
    from squaring `2T-Til=sqrt(...)`). Branch selection mirrors
    `cal_air_temp`'s own `T>253.0` (not `t_lin>253.0` -- the correct
    `til` root reproduces `T` exactly by construction, making the two
    conditions equivalent here)."""
    l_e = float(AmpsConst.L_e)
    l_s = float(AmpsConst.L_s)
    c_pa = float(AmpsConst.C_pa)
    heat = l_e * qr + l_s * qi
    til_lin = t / (1.0 + heat / (c_pa * 253.0))
    denom = t + heat / c_pa
    with np.errstate(divide="ignore", invalid="ignore"):
        til_quad = t * t / np.where(denom > 0.0, denom, 1.0)
    return np.where(t > 253.0, til_quad, til_lin)


def _diag_mes_rc_and_qr0(
    liquid: LiquidState, qvv: np.ndarray, moist_denv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`mes_rc`/`qr_0`, DUPLICATED (not imported -- `core/` must not depend
    on `implementations/`) from `implementations/warm_loop.py`'s
    `_update_mesrc_warm`/`_rain_specific_humidity`, collapsed to the 3
    reachable values (0=no water,1=vapor only,2=rain) since this driver's
    own inputs carry no ice group either. Matching this module's OWN
    `_safe_div`-duplication precedent from `core/liquid_diag.py` for small
    module-private helpers."""
    lp = LiquidPPV
    rmt = liquid.values[lp.rmt_q.py_idx]
    rmat = liquid.values[lp.rmat_q.py_idx]
    m_v = qvv * moist_denv
    m_tr = np.sum(rmt, axis=(0, 1))
    mes_rc = np.where(m_v + m_tr <= 0.0, 0, np.where(m_tr > 0.0, 2, 1)).astype(np.int64)
    qr_0 = np.sum(np.maximum(0.0, rmt - rmat), axis=(0, 1))
    qr_0 = np.where(mes_rc == 2, qr_0, 0.0)
    return mes_rc, qr_0


# ---------------------------------------------------------------------------
# CCN critical-supersaturation precompute (G2 section 1c), simplified per
# module docstring item 1: `AerosolState`'s own bins stand in directly for
# `N_act`/`M_act`.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _CcnPrecompute:
    """Per (aerosol bin, CCN-eligible category, point) CCN diagnostics,
    G2 section 1c."""

    noccn0: np.ndarray  # bool (nbins_a, ncat_ccn, npoints); True == "activates to liquid"
    n_act: np.ndarray  # (nbins_a, ncat_ccn, npoints)
    m_act: np.ndarray  # (nbins_a, ncat_ccn, npoints)
    mean_mass_grown: np.ndarray  # (nbins_a, ncat_ccn, npoints); Fortran `mean_mass(i,ica,n)`
    eps_map: np.ndarray  # (nbins_a, ncat_ccn, npoints); soluble mass fraction of this bin
    s_c: (
        np.ndarray
    )  # (nbins_a, ncat_ccn, npoints); Koehler critical supersaturation, `s_c(i,ica,n)`


def _ccn_precompute(  # noqa: PLR0915 -- single verbatim G2 section 1c transcription
    aerosol: AerosolState,
    ccn_cat_indices: tuple[int, ...],
    *,
    config: AmpsConfig,
    luts: AmpsLuts,
    icycle_active: np.ndarray,
    sw: np.ndarray,
    t: np.ndarray,
    sig_wa: np.ndarray,
    liquid_binb1: float,
) -> _CcnPrecompute:
    """G2 section 1c's Koehler activation test: the `N_bin_a=10`-point
    Gauss-Hermite mass-quadrature partition of ONE bulk lognormal aerosol
    population per category (`ga(ica)%MS(1,n)`, i.e. `AerosolState` bin
    INDEX 0 for each category -- the Fortran's own aerosol Group carries
    only a single lognormal-summary bin per category; `AerosolState`'s own
    bin axis beyond index 0 is unused here, matching that Fortran
    convention) into `N_act(i,ica,n)`/`M_act(i,ica,n)`, `i=0..N_BIN_A-1`,
    followed by the per-quadrature-bin Koehler activation test (Part 1's
    `kohler_critical_radius`, verbatim).

    `p(1)` (geometric mean radius)/`p(2)` (ln-sigma) -- the category's
    lognormal shape, read directly (`diag_pardis_ap`,
    `class_Mass_Bin.F90:1743-1789`, named nowhere in G2): this uses the
    `flagp` IN `{1,2,-1,-2,-3}` ("fixed standard deviation") branch --
    `p(1)=exp((ln(mean_mass/coef4pi3/den)-4.5*ap_lnsig^2)/3)`,
    `p(2)=ap_lnsig(icat)` -- an ASSUMPTION documented here since G2 itself
    never names `diag_pardis_ap` or its own `flagp` dummy argument (out of
    G2's own scope, read purely to resolve this driver's own precompute):
    `config.flagp_a` defaults to (and cloudlab sets) `2`, and cloudlab's
    own `config.ap_lnsig`/`config.ap_mean` are BOTH populated with
    distinct, physically-plausible per-category values -- if `flagp` were
    instead in `{3,-4,-5,-6}` ("fixed mean radius", `p(1)=ap_mean(icat)`
    directly), `ap_lnsig` would be irrelevant, which its distinct nonzero
    cloudlab values argue against. `mean_mass`/`den` are the category's
    OWN prognostic bulk state (`AerosolState` bin 0's `amt_q/acon_q`, and
    `den_ap` from the soluble-fraction mixing formula already used
    elsewhere in this module), not re-derived.

    `interp_data1d_lut_big(luts.snrml, ...)` (`_interp_data1d_lut_big`
    above) integrates the MASS-weighted quantile shift (`xi=zn-3*p(2)`,
    the lognormal 3rd-moment shift identity) to get each quadrature bin's
    mass fraction `fact_m`, verbatim (`mod_amps_core.F90:6595-6614`):

        xi2=zn(i+1)-3*p(2); yi2=interp_data1d_lut_big(snrml,abs(xi2))
        xi1=zn(i)-3*p(2);   yi1=interp_data1d_lut_big(snrml,abs(xi1))
        if(xi1>=0) then; fm2=yi1-yi2
        elseif(xi1<=0.and.xi2>0) then; fm2=1.0-yi1-yi2
        else; fm2=yi2-yi1; endif
        fact_m=min(max_fact,fm2)

    `ccn_cat_indices`: 0-based category indices to test (all categories
    except the dust/deposition category, `ica/=2` in G2).
    """
    coef4pi3 = float(AmpsConst.coef4pi3)
    den_w = float(AmpsConst.den_w)
    r_v = float(AmpsConst.R_v)
    m_w = float(AmpsConst.M_w)
    coef_a = 2.0 / (den_w * r_v)

    ncat_ccn = len(ccn_cat_indices)
    npoints = aerosol.npoints

    n_act = np.zeros((N_BIN_A, ncat_ccn, npoints), dtype=np.float64)
    m_act = np.zeros((N_BIN_A, ncat_ccn, npoints), dtype=np.float64)
    eps_map = np.zeros((N_BIN_A, ncat_ccn, npoints), dtype=np.float64)
    noccn0 = np.zeros((N_BIN_A, ncat_ccn, npoints), dtype=bool)
    mean_mass_grown = np.zeros((N_BIN_A, ncat_ccn, npoints), dtype=np.float64)
    s_c_all = np.zeros((N_BIN_A, ncat_ccn, npoints), dtype=np.float64)

    aa_over_t = coef_a * sig_wa / t  # (npoints,), `AA/ag%TV(n)%T`

    for k, cat in enumerate(ccn_cat_indices):
        # AerosolPPV order is (amt_q, acon_q, ams_q); bin 0 == Fortran
        # `MS(1,n)`, the category's single bulk lognormal-summary bin.
        amt1 = aerosol.values[0, 0, cat, :]
        acon1 = aerosol.values[1, 0, cat, :]
        ams1 = aerosol.values[2, 0, cat, :]

        eps_bulk = np.clip(_safe_div(ams1, amt1), 0.0, 1.0)
        eps_bulk = np.where(amt1 > 1.0e-30, eps_bulk, float(config.eps_ap[cat]))

        den_as = float(config.den_aps[cat])
        den_ai = float(config.den_api[cat])
        den_ap = _safe_div(np.full_like(eps_bulk, den_ai), 1.0 - eps_bulk * (1.0 - den_ai / den_as))

        mean_mass_bulk = _safe_div(amt1, acon1)
        ap_lnsig_cat = float(config.ap_lnsig[cat])

        with np.errstate(divide="ignore", invalid="ignore"):
            log_arg = _safe_div(mean_mass_bulk, coef4pi3 * den_ap)
            p1 = np.where(
                log_arg > 0.0,
                np.exp(
                    (np.log(np.where(log_arg > 0.0, log_arg, 1.0)) - 4.5 * ap_lnsig_cat**2) / 3.0
                ),
                0.0,
            )
            dum1 = coef4pi3 * den_ap * np.exp(ap_lnsig_cat) * p1**3

        eligible_bulk = (
            icycle_active
            & (sw > 0.0)
            & (acon1 >= NLMT)
            & (amt1 >= MLMT)
            & (eps_bulk >= MIN_EPS_CCN)
        )
        del dum1  # mass-bin-boundary `binba`, computed for G2 fidelity but unused downstream (see docstring)

        for i in range(N_BIN_A):
            n_act_i = FACT_C_QUADRATURE[i] * acon1

            xi2 = ZN_QUADRATURE[i + 1] - 3.0 * ap_lnsig_cat
            xi1 = ZN_QUADRATURE[i] - 3.0 * ap_lnsig_cat
            yi2 = _interp_data1d_lut_big(luts.snrml, np.abs(xi2))
            yi1 = _interp_data1d_lut_big(luts.snrml, np.abs(xi1))
            if xi1 >= 0.0:
                fm2 = yi1 - yi2
            elif xi2 > 0.0:
                fm2 = 1.0 - yi1 - yi2
            else:
                fm2 = yi2 - yi1
            fact_m = min(MAX_FACT, float(fm2))
            m_act_i = fact_m * mean_mass_bulk * acon1

            valid_i = eligible_bulk & (n_act_i >= NLMT) & (m_act_i >= MLMT)

            with np.errstate(divide="ignore", invalid="ignore"):
                r_n_micron = 1.0e4 * (
                    _safe_div(m_act_i, n_act_i) / np.where(den_ap > 0.0, den_ap, 1.0) / coef4pi3
                ) ** (1.0 / 3.0)
                sb = (
                    float(config.nu_aps[cat])
                    * eps_bulk
                    * m_w
                    * den_ap
                    / (float(config.M_aps[cat]) * den_w)
                    * float(config.phi_aps[cat])
                )
                beta = 0.5
                r_n_cm = r_n_micron * 1.0e-4
                rd_c_micron = 1.0e4 * kohler_critical_radius(aa_over_t, sb, beta, r_n_cm)
                s_c = (
                    np.exp(
                        aa_over_t * 1.0e4 / np.where(rd_c_micron != 0.0, rd_c_micron, 1.0)
                        - sb
                        * r_n_micron ** (2.0 * (1.0 + beta))
                        / np.where(
                            rd_c_micron**3 - r_n_micron**3 != 0.0,
                            rd_c_micron**3 - r_n_micron**3,
                            1.0,
                        )
                    )
                    - 1.0
                )
                activates = (
                    valid_i & (s_c <= sw) & (r_n_micron <= DRIVER_MAX_R_N) & (s_c <= SW_ALLOW)
                )

                mean_mass_ap = _safe_div(m_act_i, n_act_i)
                grown = np.maximum(liquid_binb1 * 1.05, mean_mass_ap * 1.05)

            n_act[i, k, :] = np.where(activates, n_act_i, 0.0)
            m_act[i, k, :] = np.where(activates, m_act_i, 0.0)
            eps_map[i, k, :] = eps_bulk
            noccn0[i, k, :] = activates
            mean_mass_grown[i, k, :] = np.where(activates, grown, 0.0)
            s_c_all[i, k, :] = s_c

    return _CcnPrecompute(
        noccn0=noccn0,
        n_act=n_act,
        m_act=m_act,
        mean_mass_grown=mean_mass_grown,
        eps_map=eps_map,
        s_c=s_c_all,
    )


def _all_activated_supersaturation(
    used_ma_act: np.ndarray, p: np.ndarray, rv_sat: np.ndarray, den: np.ndarray, e_satw: np.ndarray
) -> np.ndarray:
    """`sw_allact` (G2 section 1f, read directly, `mod_amps_core.F90:
    7055-7057`), verbatim:

        e_n=P*(rv_sat(1)+used_Ma_act/den)/(Rdvchiarui+(rv_sat(1)+used_Ma_act/den))
        sw_allact=e_n/e_sat(1)-1.0
    """
    rdv = float(AmpsConst.Rdvchiarui)
    rv_eff = rv_sat + used_ma_act / den
    e_n = p * rv_eff / (rdv + rv_eff)
    return e_n / e_satw - 1.0


# ---------------------------------------------------------------------------
# DHF / deposition-nucleation precompute -- genuine missing prerequisites
# (module docstring item 4): inert no-op unless the caller opts into the
# placeholder AND the branch is genuinely reachable.
# ---------------------------------------------------------------------------


def _dhf_precompute(
    npoints: int, *, iflg_dhf: int, allow_dhf_placeholder: bool
) -> dict[str, np.ndarray]:
    """`DHF_IF1` (G2 section 1d / section 4) -- see module docstring item
    4. Returns the box-level fields `ActivationBoxState` needs
    (`s_c_dhfmin`, `ds_alldhf`, `used_ma_dhf`, `akk_lmt_dhf`), all at their
    Fortran pre-loop-init ("DHF never fires") values."""
    if iflg_dhf > 0 and not allow_dhf_placeholder:
        raise NotImplementedError(
            "cal_aptact_var8_kc04dep's DHF_IF1 precompute (deliquescence-heterogeneous "
            "freezing, Savre & Ekman 2014 40-point Gauss-quadrature contact-angle-PDF "
            "integral, mod_amps_core.F90:6707-6908) is a genuine missing prerequisite -- "
            "not fully quoted by G2 (docs/superpowers/facts/m2/activation.md) and not "
            "ported anywhere in this codebase (see core/activation.py's Part 2 module "
            "docstring item 4). Pass allow_dhf_placeholder=True to treat DHF as "
            "inactive (no ice bins ever receive DHF-frozen mass) for the affected call."
        )
    return {
        "s_c_dhfmin": np.full(npoints, S_C_DHFMIN_INIT, dtype=np.float64),
        "ds_alldhf": np.zeros(npoints, dtype=np.float64),
        "used_ma_dhf": np.zeros(npoints, dtype=np.float64),
        "akk_lmt_dhf": np.ones(npoints, dtype=np.float64),
    }


def _deposition_precompute(
    npoints: int, *, iflg_dep: int, has_dust_category: bool, allow_dep_placeholder: bool
) -> dict[str, np.ndarray]:
    """`Dep_IF1` (G2 section 1e) -- see module docstring item 4. Only
    genuinely reachable when a dust (category-2) aerosol population is
    actually present (`has_dust_category`); otherwise a vacuous no-op
    regardless of `iflg_dep` (deposition nucleation is fundamentally a
    dust-aerosol process -- with no dust category in the state, there is
    nothing to nucleate from)."""
    if iflg_dep > 0 and has_dust_category and not allow_dep_placeholder:
        raise NotImplementedError(
            "cal_aptact_var8_kc04dep's Dep_IF1 precompute (classical-nucleation-theory "
            "40-point Gauss-quadrature deposition-freezing integral, mod_amps_core.F90:"
            "6912-7035) is a genuine missing prerequisite -- not fully quoted by G2 and "
            "not ported anywhere in this codebase (see core/activation.py's Part 2 module "
            "docstring item 4). Pass allow_dep_placeholder=True to treat deposition "
            "nucleation as inactive for the affected call."
        )
    return {
        "used_ma_dep": np.zeros(npoints, dtype=np.float64),
        "si_alldep": np.zeros(npoints, dtype=np.float64),
        "aerosol_dep_con": np.zeros(npoints, dtype=np.float64),
        "aerosol_dep_mass": np.zeros(npoints, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Shared condensate diagnosis -- the common first half of `func_vec`'s own
# body (`mod_amps_core.F90:9284-9298`), reused by BOTH the backward-Euler
# loop (G2 section 1h, which calls `func_liqvap_vec`/`func_icevap_vec`
# directly, not through `func_vec`) and this driver's own zbrent
# qr_b/qi_b/T recovery (G2 section 1i: `qr_b(n)=qr2(n); qi_b(n)=qi2(n)`).
# ---------------------------------------------------------------------------


def _diagnose_condensate(  # noqa: PLR0917
    x: np.ndarray,
    y: np.ndarray,
    box: ActivationBoxState,
    liq: LiquidBinState,
    ice: IceBinState,
    flags: ActivationFlags,
    estbar: np.ndarray,
    esitbar: np.ndarray,
    *,
    allow_shed_placeholder: bool,
) -> tuple[np.ndarray, np.ndarray]:
    liqvap = func_liqvap_vec(x, box, liq, flagp_r=flags.flagp_r, dt=flags.dt)
    icevap = func_icevap_vec(
        x, y, box, ice, flags, estbar, esitbar, allow_shed_placeholder=allow_shed_placeholder
    )
    with np.errstate(all="ignore"):
        trans_mi = icevap.loss_mi_mlt - np.minimum(
            liqvap.liq_left, box.gain_mi_rim + box.gain_mi_frn
        )
        qr = np.maximum(
            0.0, box.qr_0 + (liqvap.used_mr_act + liqvap.used_mr_vap + trans_mi) / box.den
        )
        qi = np.maximum(
            0.0,
            box.qi_0
            + (icevap.used_mi_vap + icevap.used_mi_vapliq + icevap.used_mi_act - trans_mi)
            / box.den,
        )
    return qr, qi


def _self_consistent_condensate(  # noqa: PLR0917
    x: np.ndarray,
    y: np.ndarray,
    box: ActivationBoxState,
    liq: LiquidBinState,
    ice: IceBinState,
    flags: ActivationFlags,
    estbar: np.ndarray,
    esitbar: np.ndarray,
    *,
    allow_shed_placeholder: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`func_vec`'s FULL body up through its `iswitch==1` second pass
    (`mod_amps_core.F90:9284-9298,9302-9313`): diagnose condensate at the
    trial `(x, y)`, re-diagnose the self-consistent `(x_n, y_n)` from that,
    then diagnose condensate AGAIN at `(x_n, y_n)` -- Fortran's own
    `qr2`/`qi2`. Needed here (not just `func_vec`'s own `.residual`)
    because `zbrent_act_vec`'s host-scope `qr2`/`qi2`/`y_n` side effects
    (its LAST internal `func_vec` call, at the converged root) have no
    Python equivalent -- G2 section 1i's `qr_b(n)=qr2(n); qi_b(n)=qi2(n)`
    is reproduced here as an EXPLICIT recovery call at the converged
    root, not a return-value `zbrent_act_vec` doesn't have. Returns
    `(x_n, y_n, qr2, qi2)`."""
    qr, qi = _diagnose_condensate(
        x, y, box, liq, ice, flags, estbar, esitbar, allow_shed_placeholder=allow_shed_placeholder
    )
    y_n = cal_air_temp(box.til, qr, qi)
    with np.errstate(all="ignore"):
        e_satw = thermo.esat_lk(1, y_n, estbar, esitbar)
        qv_n = np.maximum(0.0, box.qtp - qr - qi)
        x_n = box.pressure * qv_n / (float(AmpsConst.Rdvchiarui) + qv_n) / e_satw - 1.0
    qr2, qi2 = _diagnose_condensate(
        x_n,
        y_n,
        box,
        liq,
        ice,
        flags,
        estbar,
        esitbar,
        allow_shed_placeholder=allow_shed_placeholder,
    )
    return x_n, y_n, qr2, qi2


# ---------------------------------------------------------------------------
# Backward-Euler Til-Qt vapor-advance stepping (G2 section 1h), masked
# fixed-iteration over OUTER_ITMAX=200.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _BackwardEulerResult:
    sw_b: np.ndarray
    t_a_b: np.ndarray
    imethod: np.ndarray  # int: -1 converged, 0 never touched (icycle==1), 1 -> zbrent


def _advance_vapor_backward_euler(  # noqa: PLR0917
    box: ActivationBoxState,
    liq: LiquidBinState,
    ice: IceBinState,
    flags: ActivationFlags,
    estbar: np.ndarray,
    esitbar: np.ndarray,
    *,
    icycle_active: np.ndarray,
    qv_init: np.ndarray,
    allow_shed_placeholder: bool,
) -> _BackwardEulerResult:
    """G2 section 1h, verbatim structure: masked fixed-iteration numpy
    (all lanes run the full OUTER_ITMAX=200 budget; a boolean `active`
    mask freezes lanes that converged, oscillated, or over-consumed
    water), matching this codebase's own established idiom (Part 1's
    `zbrent_act_vec`)."""
    npoints = box.sw.shape[0]
    sw_b = box.sw.copy()
    t_a_b = box.t.copy()
    qr_b = box.qr_0.copy()
    qi_b = box.qi_0.copy()
    qr_b2 = box.qr_0.copy()
    qi_b2 = box.qi_0.copy()
    qv = qv_init.copy()

    imethod = np.zeros(npoints, dtype=np.int64)
    active = icycle_active.copy()

    for iteration in range(OUTER_ITMAX):
        if not active.any():
            break

        qr, qi = _diagnose_condensate(
            sw_b,
            t_a_b,
            box,
            liq,
            ice,
            flags,
            estbar,
            esitbar,
            allow_shed_placeholder=allow_shed_placeholder,
        )
        with np.errstate(all="ignore"):
            em = qr + qi >= box.qtp

            t_a_n = cal_air_temp(box.til, qr, qi)
            e_satw = thermo.esat_lk(1, t_a_n, estbar, esitbar)
            qv_n = np.maximum(0.0, box.qtp - qr - qi)
            sw_n = box.pressure * qv_n / (float(AmpsConst.Rdvchiarui) + qv_n) / e_satw - 1.0

            dif = np.abs(_safe_div(qv_n - qv, qv))
            dif_t = np.abs(_safe_div(t_a_n - t_a_b, t_a_b))

            oscillating = (
                (qi > 1.0e-20) & (qi_b < qi) & (qi_b2 > qi_b) & (t_a_b < float(AmpsConst.T_0))
            ) | ((qr > 1.0e-20) & (qr_b < qr) & (qr_b2 > qr_b))
            oscillating = oscillating & (iteration >= 9)  # Fortran `iter>=10`, 1-based

            hand_to_zbrent = active & (oscillating | em)
            converged = active & ~hand_to_zbrent & (dif <= ALIM1) & (dif_t <= ALIM2)
            still_iterating = active & ~hand_to_zbrent & ~converged

        imethod = np.where(hand_to_zbrent, 1, imethod)
        imethod = np.where(converged, -1, imethod)

        sw_b = np.where(still_iterating, sw_n, sw_b)
        t_a_b = np.where(still_iterating, t_a_n, t_a_b)
        qr_b2 = np.where(still_iterating, qr_b, qr_b2)
        qi_b2 = np.where(still_iterating, qi_b, qi_b2)
        qr_b = np.where(still_iterating, qr, qr_b)
        qi_b = np.where(still_iterating, qi, qi_b)
        qv = np.where(still_iterating, qv_n, qv)
        # `converged` lanes take the Fortran's own end-of-branch reassignment
        # (`sw_n(n)=sw_b(n); ...`, a no-op relative to their pre-branch sw_b/T_a_b).
        sw_b = np.where(converged, sw_b, sw_b)
        t_a_b = np.where(converged, t_a_b, t_a_b)

        active = still_iterating

    # Remaining unconverged boxes are handed to zbrent (G2 lines 7261-7267).
    imethod = np.where(icycle_active & (imethod == 0), 1, imethod)

    return _BackwardEulerResult(sw_b=sw_b, t_a_b=t_a_b, imethod=imethod)


# ---------------------------------------------------------------------------
# add_simple_vec (mod_amps_core.F90:15452-15577, class_Mass_Bin.F90 sibling
# file), verbatim for the liquid (token==2) branch -- see module docstring
# item 3 for why this is used for ALL activated-droplet placement.
# ---------------------------------------------------------------------------


def _add_simple(  # noqa: PLR0917
    binb: np.ndarray,
    np_placed: np.ndarray,
    mp_placed: np.ndarray,
    ratio_mp: np.ndarray,
    valid: np.ndarray,
    new_n: np.ndarray,
    new_m: np.ndarray,
) -> None:
    """`add_simple_vec`, liquid (`g%token==2`) branch: `g%N_nonmass==0` for
    liquid (`LiquidPPV` has no axis fields), so the `new_Q` non-mass update
    is unconditionally a no-op and is not ported. Mutates `new_n`
    `(nbins_liq, npoints)` and `new_m` `(nbins_liq, npoints, 4)` IN PLACE
    (accumulating across repeated calls, matching the literal Fortran's own
    `category_loop1` -- `new_N`/`new_M` are zeroed ONCE before the category
    loop, then each category's `add_simple_vec` call accumulates into
    them)."""
    nbins_liq = new_n.shape[0]
    n_source = np_placed.shape[0]
    for i in range(n_source):
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_mass = _safe_div(mp_placed[i], np_placed[i])
        ok = valid[i] & (np_placed[i] > 1.0e-25) & (mp_placed[i] > 1.0e-25)
        if not ok.any():
            continue

        j0 = np.searchsorted(binb, mean_mass, side="left") - 1
        j0 = np.clip(j0, 0, nbins_liq - 1)

        n_clamped = np.maximum(
            mp_placed[i] / 0.99 / binb[j0 + 1],
            np.minimum(mp_placed[i] / 1.01 / binb[j0], np_placed[i]),
        )

        points = np.nonzero(ok)[0]
        j_ok = j0[points]
        np.add.at(new_n, (j_ok, points), n_clamped[points])
        np.add.at(new_m, (j_ok, points, 0), mp_placed[i][points])
        for jj_idx in range(3):
            np.add.at(
                new_m, (j_ok, points, 1 + jj_idx), (mp_placed[i] * ratio_mp[i, :, jj_idx])[points]
            )


# ---------------------------------------------------------------------------
# activate_and_advance_vapor -- the M2a Task 4 deliverable.
# ---------------------------------------------------------------------------


def activate_and_advance_vapor(  # noqa: PLR0915, PLR0917 -- single driver, many G2 sub-steps
    liquid: LiquidState,
    aerosol: AerosolState,
    thermo_state: ThermoState,
    config: AmpsConfig,
    dt_vp: float,
    luts: AmpsLuts,
    diag: LiquidDiag,
    *,
    allow_shed_placeholder: bool = False,
    allow_dhf_placeholder: bool = False,
    allow_dep_placeholder: bool = False,
) -> tuple[LiquidState, AerosolState, ThermoState]:
    """`cal_aptact_var8_kc04dep`, the outer driver (G2 sections 1a-1n):
    activate dry aerosol into liquid droplets (Koehler), advance vapor
    over `dt_vp` via the backward-Euler Til-Qt loop + Brent
    supersaturation solve (G2 section 1h/1i/1j, Part 1 primitives), place
    activated droplets into liquid bins (`add_simple_vec`, G2 section 1k),
    and finalize the liquid/ice-phase thermo diagnostics (G2 section 1n).

    `luts.snrml` (the standard-normal survival-function LUT) drives the
    CCN mass-quadrature's `interp_data1d_lut_big` calls -- see
    `_ccn_precompute`'s own docstring.

    The DHF (`config.ice_nucleation_dhf`) and classical-CNT deposition
    (`config.ice_nucleation_deposition`) branches are genuine missing
    prerequisites -- see module docstring item 4 and
    `allow_dhf_placeholder`/`allow_dep_placeholder`.

    Returns `(liquid', aerosol', thermo')`: `aerosol'` and `thermo'` (T_n,
    s_v_n) are updated; the aerosol-mass sink from activation
    (`ga(ica)%MS(1,n)%dcondt/dmassdt`, G2 section 1k) is applied to
    `aerosol'`.
    """
    estbar, esitbar = thermo.make_esat_tables()

    npoints = liquid.npoints
    t_0 = float(AmpsConst.T_0)

    p = get_thermo_prop(thermo_state, ThermoProp.ptotv)
    t = get_thermo_prop(thermo_state, ThermoProp.tv)
    den = get_thermo_prop(thermo_state, ThermoProp.moist_denv)
    qvv = get_thermo_prop(thermo_state, ThermoProp.qvv)

    mes_rc, qr_0 = _diag_mes_rc_and_qr0(liquid, qvv, den)
    qi_0 = np.zeros(npoints, dtype=np.float64)  # no ice group, see module docstring item 4
    qtp = qvv + qr_0 + qi_0
    sw = _liquid_supersaturation(p, qvv, t, estbar, esitbar)
    si = _ice_supersaturation(sw, t, estbar, esitbar)

    # ---- section 1a: grid-box skip mask -----------------------------------
    flagp_r = int(config.flagp_r)
    flagp_s = int(config.flagp_s)
    iflg_inuc = int(config.ice_nucleation_master)
    iflg_dep = int(config.ice_nucleation_deposition)
    iflg_dhf = int(config.ice_nucleation_dhf)

    liquid_eligible = (flagp_r > 0) & (sw > 0.0)
    ice_eligible = (
        (flagp_s > 0)
        & (iflg_inuc > 0)
        & (((iflg_dep > 0) | (iflg_dhf > 0)) & (si > 0.0) & (t < t_0))
    )
    icycle_active = (mes_rc != 0) & np.where(
        mes_rc == 1, (qtp > 1.0e-20) & (liquid_eligible | ice_eligible), True
    )

    if not icycle_active.any():
        return liquid, aerosol, thermo_state

    # ---- section 1b: per-box init ------------------------------------------
    e_satw0 = thermo.esat_lk(1, t, estbar, esitbar)
    rv_sat = _rv_saturation(p, t, estbar, esitbar)
    til = _invert_cal_air_temp(t, qr_0, qi_0)

    # ---- section 1c: CCN critical-supersaturation precompute --------------
    ncat = aerosol.ncat
    dust_cat_idx = 1 if ncat >= 2 else None  # Fortran category 2 (1-based) -> 0-based index 1
    ccn_cat_indices = tuple(c for c in range(ncat) if c != dust_cat_idx)
    sig_wa = thermo.sfc_tension(t)
    liquid_binb = bin_grid.make_bin_grid("liquid", liquid.nbins, nbin_h=config.nbin_h).binb

    ccn = _ccn_precompute(
        aerosol,
        ccn_cat_indices,
        config=config,
        luts=luts,
        icycle_active=icycle_active,
        sw=sw,
        t=t,
        sig_wa=sig_wa,
        liquid_binb1=float(liquid_binb[0]),
    )

    # ---- section 1f: activated totals + akk_lmt ----------------------------
    used_ma_act = np.maximum(0.0, ccn.n_act * ccn.mean_mass_grown - ccn.m_act).sum(axis=(0, 1))
    used_na_act = ccn.n_act.sum(axis=(0, 1))

    sw_allact = _all_activated_supersaturation(used_ma_act, p, rv_sat, den, e_satw0)

    dhf_fields = _dhf_precompute(
        npoints, iflg_dhf=iflg_dhf, allow_dhf_placeholder=allow_dhf_placeholder
    )
    dep_fields = _deposition_precompute(
        npoints,
        iflg_dep=iflg_dep,
        has_dust_category=dust_cat_idx is not None,
        allow_dep_placeholder=allow_dep_placeholder,
    )

    nr_0 = np.where(mes_rc == 2, liquid.values[LiquidPPV.rcon_q.py_idx].sum(axis=(0, 1)), 0.0)
    ccn_max = float(config.CCNMAX)
    akk_lmt = np.maximum(
        0.0, (np.minimum(ccn_max, used_na_act) - nr_0) / np.maximum(1.0e-30, used_na_act)
    )

    # ---- assemble ActivationBoxState / LiquidBinState / trivial IceBinState
    gain_mi_rim = np.zeros(
        npoints, dtype=np.float64
    )  # no freezing tendencies upstream (no ice group)
    gain_mi_frn = np.zeros(npoints, dtype=np.float64)

    box = ActivationBoxState(
        mes_rc=mes_rc,
        t=t,
        til=til,
        qr_0=qr_0,
        qi_0=qi_0,
        gain_mi_rim=gain_mi_rim,
        gain_mi_frn=gain_mi_frn,
        den=den,
        qtp=qtp,
        pressure=p,
        qr_b=qr_0.copy(),
        qi_b=qi_0.copy(),
        rv=rv_sat,
        sw=sw,
        used_ma_act=used_ma_act,
        akk_lmt=akk_lmt,
        sw_allact=sw_allact,
        si_alldep=dep_fields["si_alldep"],
        used_ma_dep=dep_fields["used_ma_dep"],
        aerosol_dep_con=dep_fields["aerosol_dep_con"],
        aerosol_dep_mass=dep_fields["aerosol_dep_mass"],
        s_c_dhfmin=dhf_fields["s_c_dhfmin"],
        ds_alldhf=dhf_fields["ds_alldhf"],
        akk_lmt_dhf=dhf_fields["akk_lmt_dhf"],
        used_ma_dhf=dhf_fields["used_ma_dhf"],
    )

    liq = LiquidBinState(
        con=liquid.values[LiquidPPV.rcon_q.py_idx, :, 0, :],
        mass_total=liquid.values[LiquidPPV.rmt_q.py_idx, :, 0, :],
        mass_aerosol=liquid.values[LiquidPPV.rmat_q.py_idx, :, 0, :],
        coef1=diag.vapdep_coef1,
        coef2=diag.vapdep_coef2,
        r_act=_liquid_activation_radius(liquid, config, t, sig_wa),
        mean_mass=diag.mean_mass,
        density=diag.density,
    )

    nbins_ice_trivial = 1
    zeros_ice = np.zeros((nbins_ice_trivial, npoints), dtype=np.float64)
    ice = IceBinState(
        con=zeros_ice,
        mass_total=zeros_ice,
        mass_aerosol=zeros_ice,
        mass_meltwater=zeros_ice,
        mean_mass=zeros_ice,
        coef1=zeros_ice,
        coef2=zeros_ice,
        phase2=np.ones((nbins_ice_trivial, npoints), dtype=np.int64),
        ts_a1=zeros_ice,
        ts_b11=zeros_ice,
        ts_b12=zeros_ice,
        ts_b13=zeros_ice,
        ts_d1=np.ones((nbins_ice_trivial, npoints), dtype=np.float64),
        tmax=np.full((nbins_ice_trivial, npoints), t_0, dtype=np.float64),
        tmp_prev=np.full((nbins_ice_trivial, npoints), t_0, dtype=np.float64),
        ldmassdt2=zeros_ice,
        e_l01=zeros_ice,
    )

    flags = ActivationFlags(
        flagp_r=flagp_r,
        flagp_s=flagp_s,
        iflg_inuc=iflg_inuc,
        iflg_dep=iflg_dep,
        iflg_dhf=iflg_dhf,
        level=int(config.level_comp),
        dt=dt_vp,
    )

    # ---- section 1h: backward-Euler Til-Qt loop ----------------------------
    be = _advance_vapor_backward_euler(
        box,
        liq,
        ice,
        flags,
        estbar,
        esitbar,
        icycle_active=icycle_active,
        qv_init=qvv,
        allow_shed_placeholder=allow_shed_placeholder,
    )

    final_sw = be.sw_b.copy()
    final_t = be.t_a_b.copy()

    # ---- section 1i: zbrent stage 1 (Til-Qt vapor-advance root) ------------
    needs_zbrent1 = icycle_active & (be.imethod == 1)
    if needs_zbrent1.any():
        t_a_ref1 = np.where(needs_zbrent1, t, be.t_a_b)  # T_a_o=T_a_b=ag%TV(n)%T at zbrent-1 init

        def _residual1(x: np.ndarray) -> np.ndarray:
            return func_vec(
                x,
                t_a_ref1,
                1,
                box,
                liq,
                ice,
                flags,
                estbar,
                esitbar,
                allow_shed_placeholder=allow_shed_placeholder,
            ).residual

        z1 = zbrent_act_vec(
            _residual1, np.ones(npoints, dtype=np.int64), 1, sw, t_a_ref1, estbar, esitbar
        )
        _x_n1, y_n1, qr2, qi2 = _self_consistent_condensate(
            z1.sw_n,
            t_a_ref1,
            box,
            liq,
            ice,
            flags,
            estbar,
            esitbar,
            allow_shed_placeholder=allow_shed_placeholder,
        )
        t_a_after1 = y_n1

        box = dataclasses.replace(
            box,
            qr_b=np.where(needs_zbrent1, qr2, box.qr_b),
            qi_b=np.where(needs_zbrent1, qi2, box.qi_b),
        )

        # oscillation check: func_vec(iswitch=0) at (sw_n, T_a_b=t_a_ref1).
        osc = func_vec(
            z1.sw_n,
            t_a_ref1,
            0,
            box,
            liq,
            ice,
            flags,
            estbar,
            esitbar,
            allow_shed_placeholder=allow_shed_placeholder,
        )
        oscillating1 = needs_zbrent1 & (np.abs(osc.residual) > 0.1)

        sw_n1 = np.where(oscillating1, sw, z1.sw_n)
        t_a_b1 = np.where(oscillating1, t, t_a_ref1)
        box = dataclasses.replace(
            box,
            qr_b=np.where(oscillating1, qr_0, box.qr_b),
            qi_b=np.where(oscillating1, qi_0, box.qi_b),
        )

        # sw_m for the section-1j gate: re-run func_vec(iswitch=0) at the
        # (possibly reset) state.
        osc2 = func_vec(
            sw_n1,
            t_a_b1,
            0,
            box,
            liq,
            ice,
            flags,
            estbar,
            esitbar,
            allow_shed_placeholder=allow_shed_placeholder,
        )
        sw_m1 = osc2.x_n

        final_sw = np.where(needs_zbrent1, sw_n1, final_sw)
        final_t = np.where(needs_zbrent1, np.where(oscillating1, t, t_a_after1), final_t)
    else:
        sw_m1 = final_sw.copy()

    # ---- section 1j: zbrent stage 2 (water-saturation adjustment) ---------
    iqvlmt_is_zbrent = needs_zbrent1
    sw_ref_1j = np.where(iqvlmt_is_zbrent, sw_m1, final_sw)
    needs_zbrent2 = icycle_active & (qr_0 > 1.0e-10) & (sw_ref_1j > SW_ALLOW)
    if needs_zbrent2.any():
        t_a_ref2 = np.where(needs_zbrent2, t, final_t)

        def _residual2(x: np.ndarray) -> np.ndarray:
            return func_vec(
                x,
                t_a_ref2,
                2,
                box,
                liq,
                ice,
                flags,
                estbar,
                esitbar,
                allow_shed_placeholder=allow_shed_placeholder,
            ).residual

        z2 = zbrent_act_vec(
            _residual2, np.ones(npoints, dtype=np.int64), 2, sw, t_a_ref2, estbar, esitbar
        )
        qr3, qi3 = _diagnose_condensate(
            z2.sw_n,
            t_a_ref2,
            box,
            liq,
            ice,
            flags,
            estbar,
            esitbar,
            allow_shed_placeholder=allow_shed_placeholder,
        )
        t_a_after2 = cal_air_temp(box.til, qr3, qi3)
        final_sw = np.where(needs_zbrent2, z2.sw_n, final_sw)
        final_t = np.where(needs_zbrent2, t_a_after2, final_t)

    # ---- section 1k: placement of activated droplets into liquid bins -----
    new_n = np.zeros((liquid.nbins, npoints), dtype=np.float64)
    new_m = np.zeros((liquid.nbins, npoints, 4), dtype=np.float64)  # LiquidMassIndex order

    aerosol_after = aerosol.values.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        sw_allact_safe = np.where(box.sw_allact > 1.0e-25, box.sw_allact, 1.0)
        akk = np.where(
            box.sw_allact > 1.0e-25,
            np.clip(np.minimum(box.akk_lmt, final_sw / sw_allact_safe), 0.0, 1.0),
            np.clip(box.akk_lmt, 0.0, 1.0),
        )

    for k, cat in enumerate(ccn_cat_indices):
        noccn0_k = ccn.noccn0[:, k, :]
        n_act_k = ccn.n_act[:, k, :]
        m_act_k = ccn.m_act[:, k, :]
        mean_mass_k = ccn.mean_mass_grown[:, k, :]
        eps_k = ccn.eps_map[:, k, :]

        np_placed = np.where(noccn0_k, n_act_k * akk[None, :], 0.0)
        mp_placed = np.where(noccn0_k, n_act_k * akk[None, :] * mean_mass_k, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_rmat = _safe_div(m_act_k * akk[None, :], mp_placed)
        ratio_rmas = eps_k * ratio_rmat
        ratio_rmai = np.maximum(0.0, ratio_rmat - ratio_rmas)
        ratio_mp = np.stack([ratio_rmat, ratio_rmas, ratio_rmai], axis=-1)

        _add_simple(liquid_binb, np_placed, mp_placed, ratio_mp, noccn0_k, new_n, new_m)

        # ---- aerosol mass/number sink from activation, G2 section 1k
        # verbatim (`mod_amps_core.F90:7663-7666`):
        #   ga(ica)%MS(1,n)%dcondt(pro_type)=-sum_ac(n,ica)/gr%dt
        #   ga(ica)%MS(1,n)%dmassdt(j,pro_type)=-sum_am(n,ica)*mass(j)/mass(1)/gr%dt
        # `sum_ac`/`sum_am` are POINT-LEVEL scalars per category, summed
        # over ALL N_BIN_A quadrature bins -- applied to `AerosolState` bin
        # 0 ONLY (Fortran `MS(1,n)`, this category's single real bulk bin;
        # see `_ccn_precompute`'s docstring for why the quadrature bins
        # `i` are a SYNTHETIC partition of that one bin, not distinct
        # `AerosolState` bins). `sum_am`'s per-particle-mass analogue is
        # `M_act(i,ica,n)*akk(n)` -- the ORIGINAL, ungrown aerosol mass of
        # the activated fraction, distinct from `used_Ma_act` (the VAPOR
        # consumed by post-activation growth).
        activated_num = np.where(noccn0_k, n_act_k * akk[None, :], 0.0)
        activated_mass = np.where(noccn0_k, m_act_k * akk[None, :], 0.0)
        sum_ac = activated_num.sum(axis=0)
        sum_am = activated_mass.sum(axis=0)

        amt1 = aerosol.values[0, 0, cat, :]
        ams1 = aerosol.values[2, 0, cat, :]
        ams_frac1 = _safe_div(ams1, amt1)

        aerosol_after[0, 0, cat, :] -= sum_am
        aerosol_after[2, 0, cat, :] -= sum_am * ams_frac1
        aerosol_after[1, 0, cat, :] -= sum_ac
    aerosol_after = np.maximum(0.0, aerosol_after)

    liquid_after = liquid.values.copy()
    liquid_after[LiquidPPV.rmt_q.py_idx, :, 0, :] += new_m[:, :, 0]
    liquid_after[LiquidPPV.rcon_q.py_idx, :, 0, :] += new_n
    liquid_after[LiquidPPV.rmat_q.py_idx, :, 0, :] += new_m[:, :, LiquidMassIndex.rmat.py_idx]
    liquid_after[LiquidPPV.rmas_q.py_idx, :, 0, :] += new_m[:, :, LiquidMassIndex.rmas.py_idx]

    # ---- section 1n: finalize thermo ---------------------------------------
    thermo_values = thermo_state.values.copy()
    tv_idx = list(ThermoState.PROPS).index(ThermoProp.tv)
    thermo_values[tv_idx, 0, 0, :] = np.where(icycle_active, final_t, t)
    qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
    qr_final, qi_final = _diagnose_condensate(
        final_sw,
        final_t,
        box,
        liq,
        ice,
        flags,
        estbar,
        esitbar,
        allow_shed_placeholder=allow_shed_placeholder,
    )
    del qi_final  # no ice group; qtp-qr-qi collapses to qtp-qr here
    qv_final = np.maximum(0.0, box.qtp - qr_final)
    thermo_values[qvv_idx, 0, 0, :] = np.where(icycle_active, qv_final, qvv)
    thermo_out = ThermoState(values=thermo_values)

    return LiquidState(values=liquid_after), AerosolState(values=aerosol_after), thermo_out


def _liquid_activation_radius(
    liquid: LiquidState, config: AmpsConfig, t: np.ndarray, sig_wa: np.ndarray
) -> np.ndarray:
    """`gr%MS(j,n)%r_act` (Koehler activation radius of a liquid bin's own
    aerosol content, `func_liqvap_vec`'s `cond2` shrink-below-critical-
    radius check) -- NOT exposed by `core.liquid_diag.LiquidDiag` (Task 2's
    own documented scope: "r_crt/r_act ... not in the task's named field
    list"). Recomputed here from `LiquidState`'s own aerosol-mass PPV
    slots using the SAME formula `core/liquid_diag.py`'s private
    `_den_aclen`/`_vapdep_coef` helpers already use internally (den_ap
    from soluble-fraction mixing, `r_n` from aerosol mass/density,
    `kohler_critical_radius` -- Part 1, verbatim) -- not a new formula,
    just re-deriving the ONE field Task 2 chose not to return. Category 1
    (0-based index 0), matching `core/liquid_diag.py`'s own hardcoded
    single-category convention (see that module's docstring)."""
    coef4pi3 = float(AmpsConst.coef4pi3)
    den_w = float(AmpsConst.den_w)
    r_v = float(AmpsConst.R_v)
    m_w = float(AmpsConst.M_w)

    mass_aero_total = liquid.values[LiquidPPV.rmat_q.py_idx, :, 0, :]
    mass_aero_soluble = liquid.values[LiquidPPV.rmas_q.py_idx, :, 0, :]
    con = liquid.values[LiquidPPV.rcon_q.py_idx, :, 0, :]

    eps_map = np.clip(_safe_div(mass_aero_soluble, mass_aero_total), 0.0, 1.0)
    eps_map = np.where(mass_aero_total > 1.0e-30, eps_map, float(config.eps_ap[0]))

    den_as = float(config.den_aps[0])
    den_ai = float(config.den_api[0])
    den_ap = _safe_div(np.full_like(eps_map, den_ai), 1.0 - eps_map * (1.0 - den_ai / den_as))

    map_ = _safe_div(mass_aero_total, con)  # per-particle aerosol mass
    r_n = _safe_div(map_, coef4pi3 * den_ap) ** (1.0 / 3.0)

    aa = 2.0 * sig_wa[None, :] / (r_v * t[None, :] * den_w)
    sb = (
        float(config.nu_aps[0])
        * eps_map
        * m_w
        * den_ap
        / (float(config.M_aps[0]) * den_w)
        * float(config.phi_aps[0])
    )
    beta = 0.5
    return kohler_critical_radius(np.broadcast_to(aa, r_n.shape), sb, beta, r_n)

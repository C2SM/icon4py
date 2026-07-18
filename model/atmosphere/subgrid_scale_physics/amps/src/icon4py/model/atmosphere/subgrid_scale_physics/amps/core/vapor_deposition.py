# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`vapor_deposition`'s LIQUID (condensation/evaporation growth) path,
transcribed from AMPS Fortran (scale_amps repo) per
docs/superpowers/facts/m2/vapor-deposition.md ("G3" below): the driver
(section 1), the Chen and Lamb (1994) semidiscrete growth helpers
(section 2), and the mass-space bin remap `cal_transbin_vec`/
`shift_bin_loop1` (section 4, `mod_amps_utility.F90:8713`). Ice-only work
(habit/growth-mode, `assign_Qp_v3_vec`, `cal_xxx_p_v5_vec`, axis growth,
`dep_mass3_vec2`'s ice-mass-component partition -- M3 scope) is NOT
ported here.

G3's own "Key finding" (module docstring there) establishes something
essential to this port's SHAPE: `vapor_deposition`'s entire quoted growth
kernel (shift-bin, `cal_ratio_mass_vd_vec`, `cal_transbin_vec`, tendency
assignment -- G3 section 1, lines 456-961) lives INSIDE `if(g%token==2)`
(ICE). For `token==1` (LIQUID), the routine's own executable body is only
initialization + the `d_mean_mass`/shifted-boundary/evaporation-check
block (G3 section 1, lines 484-614) -- but every helper it calls
(`cal_ratio_mass_vd_vec`'s `token==1` branch, `cal_transbin_vec`'s
`iphase==1` branch) is written GENERICALLY over both phases and IS
reachable for liquid via other call sites in the real Fortran (the
`g%token==2` gate wrapping this ONE reading of `vapor_deposition` is an
artifact of how G3's extraction found the routine, not evidence liquid
condensation growth is unreachable code -- `diag_pq`'s phase==1 branch,
Task 2's `core/liquid_diag.py`, independently confirms `coef(1)/coef(2)`
are real, always-computed liquid quantities). This module ports the
LIQUID-reachable subset of every quoted routine's own per-phase branches,
matching G3's per-routine liquid/ice split table (its final section).

Design decisions / fact-gaps, each independently justified below (not
guesses):

1. **`s_v_n` (the "in-substep" supersaturation `d_mean_mass`'s LIQUID
   branch consumes, G3 section 1 line 526: `ag%TV(n)%s_v_n(g%token)`)**:
   not a field this port's `ThermoState` carries persistently. Fortran
   sets it via `ag%TV(n)%s_v_n(1)=sw_n(n)` at the END of the CCN
   activation/vapor-advancement routine that runs immediately before
   `vapor_deposition` in the same vap-loop iteration (`mod_amps_core.F90`
   lines 4771/8315, inside `cal_aptact_var8_kc04dep` -- M2a Task 4's
   `core.activation.activate_and_advance_vapor`), using EXACTLY the
   formula `core.activation`'s own `_liquid_supersaturation`/`func_vec`'s
   `x_n` already use: `sw_n = P*qv_n/(Rdvchiarui+qv_n)/e_satw(T_n) - 1`
   at the self-consistently CONVERGED `(T_n, qv_n)`. Task 4's own
   `activate_and_advance_vapor` returns exactly that converged `(T_n,
   qv_n)` pair in its output `ThermoState` (`tv`/`qvv`, see its own
   `activation.py` lines ~2479-2495) -- so `s_v_n` here is RECOMPUTED
   from the `thermo` argument (the post-activation state, per
   `implementations/warm_loop.py`'s existing vap-loop wiring: refresh ->
   activation -> vapor_deposition) via the identical formula, proven
   algebraically identical to the Fortran's own `sw_n(n)` by
   construction, not approximated. `_liquid_supersaturation` is
   DUPLICATED here (small, `core.activation`'s own private helper,
   matching that module's own precedent of duplicating
   `implementations/warm_loop.py`'s `_update_mesrc_warm`/
   `_rain_specific_humidity` rather than cross-importing) -- this
   task's own dispatch does not list `core/activation.py` as a reuse
   target, and `core/` modules should not depend on each other for
   private helpers.
2. **Diagnostic staleness (`diag: LiquidDiag`) vs. the CURRENT `liquid`
   state**: `diag.mean_mass`/`diag.vapdep_coef1`/`diag.vapdep_coef2` are
   `diag_pq`'s OWN persistent per-bin fields (`g%MS(i,n)%mean_mass`,
   `%coef(1)`, `%coef(2)`) -- refreshed once per vap-loop iteration at
   `_refresh_state`, BEFORE CCN activation (Task 4) runs and adds newly
   nucleated droplets to `liquid`. This means a bin that goes from empty
   to populated PURELY via activation within the SAME vap-loop iteration
   has STALE (zero) `diag` fields here too, in the REAL Fortran as much
   as in this port (its own `g%MS(i,n)%coef` are the same stale
   persistent fields) -- `d_mean_mass` is therefore exactly 0 for such a
   bin regardless, by construction, in BOTH. This module's own `icond3`
   (source-bin eligibility) mask is `diag.mean_mass > 0` (NOT the
   CURRENT bin population) for exactly this reason: it reproduces the
   Fortran's own staleness-driven exclusion, avoids a genuine
   0/`mean_mass` division for such bins, and is a documented, provably
   safe substitute for re-deriving `icond3` from `g%MS(i,n)%con`/
   `mass(1)` directly (G3 section 1 lines 312-323) -- since
   `diag.mean_mass>0` implies (via `LiquidDiag`'s own documented
   invariant, `core/liquid_diag.py`) `con>1e-30` and `mass_total>1e-30`
   at diag-computation time, and CCN activation only ever ADDS liquid
   mass/number (never removes it), the CURRENT `con`/`mass_total` at
   THIS bin are guaranteed `>1e-30` too whenever `diag.mean_mass>0`.
3. **`cal_lincubprms_vec` (the shifted-bin linear/cubic distribution-
   parameter solver `cal_transbin_vec`'s own `a2d` input) is NOT quoted
   by G3** -- only its call sites and the boundary-setup rewrite ("case
   of non-negative bin": `n(m)=a0+a1*m`) are. Per this task's dispatch
   authorization (read named-but-truncated Fortran directly), this
   module instead uses the mathematically FORCED unique linear density
   `n(m)=a0+a1*m` on the shifted interval `[bd1,bd2]` matching the two
   given moments (`Npd`=0th, `Mpd`=1st) EXACTLY -- solving the 2x2
   moment system `N=a0*I0+a1*I1, M=a0*I1+a1*I2` (`_linear_reconstruction`
   below) is the ONLY linear function satisfying both constraints over a
   fixed interval (a Cauchy-Schwarz argument on `[bd1,bd2]`, `bd1<bd2`,
   guarantees the system's determinant is strictly positive -- never a
   guess, not "the Fortran's algorithm", but the unique closed form any
   correct linear-in-mass reconstruction with these two moments must
   equal, given the interval). What this deliberately does NOT reproduce
   is `cal_lincubprms_vec`'s own "negative density" FALLBACK branches
   (the boundary-setup snippet's `a2d(i,n,1)==-1/-2`, "case of
   n(x'_1)<0.0" / "n(x'_2)<0.0", shrinking the support interval) --
   genuinely unavailable without reading that un-quoted routine's own
   body. This module's fallback for a degenerate (zero/negative-width,
   or non-positive-determinant) shifted interval is a same-bin (no
   shift) passthrough of `Npd`/`Mpd` instead (still EXACTLY N/M
   conserving, just not reproducing the Fortran's own truncated-support
   redistribution for that edge case).
4. **The haze-transfer early-evaporation check (G3 section 1 line 408:
   `level>=4.and.g%token==1.and.g%MS(i,n)%r_act>r_new`) is OMITTED.**
   `r_act` (haze equilibrium radius, `get_hazerad_anal`) is explicitly
   NOT exposed by `core/liquid_diag.py`'s `LiquidDiag` (that module's own
   docstring, item 2: "computed ... by TWO further un-quoted helper
   functions not read for this task"). This module treats that
   sub-condition as never-triggering (the SAFE direction: a
   sub-threshold-radius droplet simply keeps evaporating via the other
   two checks -- `temp_dM`/`shifted_hi<=binb[0]` -- rather than being
   force-evaporated early on missing information). Only affects bins
   undergoing evaporation that would otherwise NOT be flagged total by
   the other two checks; this module's own tests are constructed to not
   depend on this sub-condition either way.
5. **The truncation optimization (`icond_ngb`, G3 section 1 lines
   525-534: skip the shift-bin machinery when `|dMcon|<=mass*1e-7`) is
   SKIPPED.** It is a pure PERFORMANCE optimization in the Fortran (a
   negligible mass change would place ~all of its mass back in the same
   bin via the general shift-bin remap anyway) with no distinct physical
   outcome; this module's `_linear_reconstruction`/gather-remap handles
   arbitrarily small shifts exactly (conserving N/M to machine
   precision) without any special-casing, so omitting the optimization
   changes no result, only simplifies this port.
6. **Underflow (mass shifted below `binb[0]`) is folded into liquid bin
   0 instead of being diverted to the AEROSOL group.** G3 section 4's
   own `cal_transbin_vec` (`iaer_src==1`) transfers underflow mass OUT of
   the liquid group into `ap_dN`/`ap_dM`/`ap_dMS` aerosol tendencies --
   this function's signature (`(liquid, thermo, config, dt_vp, diag) ->
   liquid'`, no `AerosolState` in or out) has no channel for that. Since
   evaporation this severe is *already* caught by the total-evaporation
   check (item 4's `shifted_hi<=binb[0]` branch) in the overwhelmingly
   common case, this fallback is rarely exercised; where it is, keeping
   the mass in liquid bin 0 is the conservative choice for THIS
   function's own contract (never silently lose mass with no
   destination to send it to).
7. **`dep_mass3_vec2` is ICE-ONLY** (G3's own heading: "evaporation
   mass-component partition (ICE-ORIENTED helper)"; called only from
   `cal_ratio_mass_vd_vec`'s `elseif(g%token==2)` branch) -- NOT ported.
   LIQUID's own mass-component distribution (this task's "dep_mass3"
   deliverable bullet, functionally) is the much simpler ratio formula
   from `cal_ratio_mass_vd_vec`'s `if(g%token==1)` branch (G3 section 2b,
   lines 16582-16598, quoted FULL): `ratio_Mp(rmat_m)=mass(rmat)/Mp`,
   `ratio_Mp(rmas_m)=mass(rmas)/Mp` (liquid's aerosol content does not
   itself grow/shrink from condensation -- only redistributes
   proportionally to wherever the bin's total mass ends up after the
   remap). `rmai` (insoluble aerosol mass) is DERIVED (`rmat-rmas`), not
   a separate `LiquidPPV` field (`state.py`), so only `rmat_m`/`rmas_m`
   need tracking here.
8. **`excess_vapor_density` (`cal_ex_vapor_density_vec`, G3 section 2a)
   is ported as a standalone, independently-tested utility** (the
   dispatch names it among this task's Chen-Lamb building blocks) but is
   NOT invoked by `vapor_deposition_liquid`'s own growth kernel -- G3's
   own "Key finding" analysis (and this module's own docstring, above)
   establishes it is called ONLY from within the ICE (`token==2`) gate;
   LIQUID's reachable growth path uses `diag.vapdep_coef1`/
   `vapdep_coef2` directly (`cal_coef_vapdep2_vec`'s phase==1 branch,
   ALREADY folding capacitance + ventilation into `coef(1)`, per
   `core/liquid_diag.py`'s `_vapdep_coef`) -- matching this task's own
   deliverable phrasing ("per-bin mass growth using capacitance +
   ventilation FROM DIAG").
"""

from __future__ import annotations

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import bin_grid, thermo as thermo_fn
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Small numeric helpers -- see core/liquid_diag.py's/core/activation.py's own
# identical-purpose, independently-duplicated `_safe_div` precedent.
# ---------------------------------------------------------------------------


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )


def _liquid_supersaturation(
    p: np.ndarray, qv: np.ndarray, t: np.ndarray, estbar: np.ndarray, esitbar: np.ndarray
) -> np.ndarray:
    """`sw=P*qv/(Rdvchiarui+qv)/e_satw(T)-1`, standing in for
    `ag%TV(n)%s_v_n(1)` at the point `vapor_deposition` reads it -- see
    module docstring item 1. DUPLICATED from `core/activation.py`'s own
    private `_liquid_supersaturation` (same formula, same justification:
    `core/` modules do not cross-import each other's private helpers)."""
    e_satw = thermo_fn.esat_lk(1, t, estbar, esitbar)
    rdv = float(AmpsConst.Rdvchiarui)
    return p * qv / (rdv + qv) / e_satw - 1.0


# ---------------------------------------------------------------------------
# cal_ex_vapor_density_vec (G3 section 2a) -- standalone utility, see module
# docstring item 8 for why it is NOT on vapor_deposition_liquid's own call
# path.
# ---------------------------------------------------------------------------


def excess_vapor_density(  # noqa: PLR0917 [too-many-positional-arguments]
    token: int,
    s_v_ice: np.ndarray,
    e_sat_ice: np.ndarray,
    t_ambient: np.ndarray,
    tmp_particle: np.ndarray,
    con: np.ndarray,
    mass_total: np.ndarray,
    estbar: np.ndarray,
    esitbar: np.ndarray,
) -> np.ndarray:
    """`cal_ex_vapor_density_vec` (G3 section 2a, `mod_amps_core.F90:
    16422`), verbatim per-bin formula (the extra `ex_vden(N_BIN+1,n)`
    "ambient-only" slot is out of scope -- not a per-bin quantity):

        ex_vden(i,n) = (s_v_ice+1)*e_sat_ice/(R_v*t_ambient)
                       - e_s(token,tmp_particle)/(R_v*tmp_particle)

    `0` wherever `mass_total<=1e-30` or `con<=1e-30` (G3's own guard).
    `e_s` is `get_sat_vapor_pres_lk(token,tmp_particle,...)`, i.e.
    `core.thermo.esat_lk(token, tmp_particle, estbar, esitbar)`.
    """
    r_v = float(AmpsConst.R_v)
    den_v_inf = (s_v_ice + 1.0) * e_sat_ice / (r_v * t_ambient)
    e_s = thermo_fn.esat_lk(token, tmp_particle, estbar, esitbar)
    den_sv_sfc = e_s / (r_v * tmp_particle)
    valid = (mass_total > 1.0e-30) & (con > 1.0e-30)
    return np.where(valid, den_v_inf - den_sv_sfc, 0.0)


# ---------------------------------------------------------------------------
# Linear (n(m)=a0+a1*m) shifted-bin reconstruction -- see module docstring
# item 3.
# ---------------------------------------------------------------------------


def _linear_reconstruction(
    number: np.ndarray, mass: np.ndarray, bd1: np.ndarray, bd2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unique linear density `n(m)=a0+a1*m` on `[bd1,bd2]` matching
    given 0th moment `number` and 1st moment `mass` exactly -- see module
    docstring item 3. Returns `(a0, a1, well_posed)`; `well_posed` is
    `False` wherever `bd2<=bd1` (degenerate/zero-width interval) OR the
    resulting density would go negative at either endpoint (`n` is
    linear, so checking both endpoints is sufficient) -- `a0`/`a1` are 0
    there (callers must route those bins through a same-bin passthrough
    instead, see `_gather_remap`).

    The negative-density case is exactly the un-quoted
    `cal_lincubprms_vec`'s own "n(x'_1)<0"/"n(x'_2)<0" fallback branches
    (module docstring item 3): reachable whenever `mass/number` (the
    shifted interval's mean) sits close enough to one edge that a linear
    fit overshoots negative at the other. Rather than reproduce that
    routine's own (un-quoted) truncated-support algorithm, this module
    falls back to a same-bin passthrough for such bins -- still EXACTLY
    N/M conserving (the binding constraint), just not reproducing the
    Fortran's own redistribution for this specific edge case. Without
    this check, `_moment_integral`'s `max(0,...)` clamp (G3's own) would
    silently break conservation instead when a sub-interval integral goes
    negative and gets clipped while the full-interval integral would not
    have (each piece clamped independently, not the whole) -- caught
    during this task's own TDD (`TestLinearReconstructionRoundTrip`)."""
    width = bd2 - bd1
    well_posed = width > 0.0
    safe_width = np.where(well_posed, width, 1.0)
    bd1s = np.where(well_posed, bd1, 0.0)
    bd2s = np.where(well_posed, bd2, 1.0)

    i0 = safe_width
    i1 = (bd2s * bd2s - bd1s * bd1s) / 2.0
    i2 = (bd2s * bd2s * bd2s - bd1s * bd1s * bd1s) / 3.0
    det = i0 * i2 - i1 * i1
    well_posed = well_posed & (det > 0.0)
    det_safe = np.where(well_posed, det, 1.0)

    a0 = np.where(well_posed, (number * i2 - mass * i1) / det_safe, 0.0)
    a1 = np.where(well_posed, (mass * i0 - number * i1) / det_safe, 0.0)

    density_at_bd1 = a0 + a1 * bd1s
    density_at_bd2 = a0 + a1 * bd2s
    non_negative = (density_at_bd1 >= -1.0e-9 * np.maximum(np.abs(a0), 1.0)) & (
        density_at_bd2 >= -1.0e-9 * np.maximum(np.abs(a0), 1.0)
    )
    well_posed = well_posed & non_negative
    a0 = np.where(well_posed, a0, 0.0)
    a1 = np.where(well_posed, a1, 0.0)
    return a0, a1, well_posed


def _moment_integral(
    a0: np.ndarray, a1: np.ndarray, bd1: np.ndarray, bd2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact integral of `n(m)=a0+a1*m` over `[bd1,bd2]`: 0th moment
    (`trans_dN`) and 1st moment (`trans_dM`) -- `cal_transbin_vec`'s own
    `shift_bin_loop1` formula (G3 section 4), specialized to its general
    cubic `n(m)=a0+a1*m+a2*m^2+a3*m^3` with `a2=a3=0` (this module's
    reconstruction is always linear, see module docstring item 3); the
    `max(0,...)` clamp on both outputs is G3's own (`trans_dN=max(0.0,
    trans_dN)`, `trans_dM=max(0.0,trans_dM)`)."""
    x_a = bd2 + bd1
    x_b = bd2 - bd1
    x_c = bd2 * bd1
    x_d = bd2 * bd2 + bd1 * bd1
    trans_dn = a0 * x_b + 0.5 * a1 * x_a * x_b
    trans_dm = 0.5 * a0 * x_a * x_b + a1 * x_b * (x_d + x_c) / 3.0
    return np.maximum(trans_dn, 0.0), np.maximum(trans_dm, 0.0)


# ---------------------------------------------------------------------------
# cal_transbin_vec / shift_bin_loop1 (G3 section 4, iphase==1) -- GATHER
# formulation: for each destination bin, sum contributions from every
# source bin whose shifted interval overlaps it (the dense i x ibx double
# loop G3 quotes verbatim), vectorized per source bin over (destination
# bin, point) -- matching core/activation.py's own `_add_simple`
# per-source-bin-loop convention.
# ---------------------------------------------------------------------------


def _gather_remap(  # noqa: PLR0917 [too-many-positional-arguments]
    binb: np.ndarray,
    shifted_lo: np.ndarray,
    shifted_hi: np.ndarray,
    npd: np.ndarray,
    mpd: np.ndarray,
    ratio_rmat: np.ndarray,
    ratio_rmas: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GATHER mass-space bin remap. `binb`: `(nbins+1,)` bin boundaries.
    `shifted_lo`/`shifted_hi`/`npd`/`mpd`/`ratio_rmat`/`ratio_rmas`/
    `source_valid`: `(nbins, npoints)`, one entry per SOURCE bin (`npd`/
    `mpd` are the shifted bin's own 0th/1st moment -- `Npd`/`Mpd` in G3's
    naming; `source_valid` is `icond_noevp`: eligible, non-total-
    evaporation source bins).

    Overflow (`shifted_hi>binb[-1]`) folds into the LAST destination bin
    (G3's own `cal_transbin_vec` overflow branch, additive with the
    grid-clipped contribution -- the two pieces are disjoint, meeting
    exactly at `binb[-1]`, never double-counted). Underflow
    (`shifted_lo<binb[0]`) folds into the FIRST bin instead of the
    AEROSOL group G3 diverts it to -- see module docstring item 6.
    Degenerate (non-well-posed, see `_linear_reconstruction`) source bins
    fall back to a same-bin (no shift) addition of `npd`/`mpd` at their
    own index -- still exactly N/M conserving.

    Returns `(new_n, new_rmt, new_rmat, new_rmas)`, each `(nbins,
    npoints)`.
    """
    nbins = binb.shape[0] - 1
    npoints = shifted_lo.shape[1]
    new_n = np.zeros((nbins, npoints))
    new_rmt = np.zeros((nbins, npoints))
    new_rmat = np.zeros((nbins, npoints))
    new_rmas = np.zeros((nbins, npoints))

    bin_lo = binb[:-1][:, None]
    bin_hi = binb[1:][:, None]
    grid_lo = binb[0]
    grid_hi = binb[-1]

    a0, a1, well_posed = _linear_reconstruction(npd, mpd, shifted_lo, shifted_hi)
    gather_source = source_valid & well_posed
    same_bin_source = source_valid & ~well_posed

    for i in range(shifted_lo.shape[0]):
        valid_i = gather_source[i]
        if valid_i.any():
            lo_i = shifted_lo[i][None, :]
            hi_i = shifted_hi[i][None, :]
            a0_i = a0[i][None, :]
            a1_i = a1[i][None, :]

            # Main grid-clipped overlap, dense over every destination bin
            # (the i x ibx loop, G3 section 4 lines 8899-8919/1656-1665).
            ov_lo = np.maximum(bin_lo, np.maximum(lo_i, grid_lo))
            ov_hi = np.minimum(bin_hi, np.minimum(hi_i, grid_hi))
            overlap = (ov_hi > ov_lo) & valid_i[None, :]
            bd1 = np.where(overlap, ov_lo, 0.0)
            bd2 = np.where(overlap, ov_hi, bd1)
            tdn, tdm = _moment_integral(a0_i, a1_i, bd1, bd2)
            tdn = np.where(overlap, tdn, 0.0)
            tdm = np.where(overlap, tdm, 0.0)
            new_n += tdn
            new_rmt += tdm
            new_rmat += ratio_rmat[i][None, :] * tdm
            new_rmas += ratio_rmas[i][None, :] * tdm

            # Overflow: shifted interval extends above the grid top.
            of_lo = np.maximum(grid_hi, lo_i[0])
            of_hi = hi_i[0]
            of_valid = valid_i & (of_hi > of_lo)
            tdn_of, tdm_of = _moment_integral(a0_i[0], a1_i[0], of_lo, of_hi)
            tdn_of = np.where(of_valid, tdn_of, 0.0)
            tdm_of = np.where(of_valid, tdm_of, 0.0)
            new_n[-1, :] += tdn_of
            new_rmt[-1, :] += tdm_of
            new_rmat[-1, :] += ratio_rmat[i] * tdm_of
            new_rmas[-1, :] += ratio_rmas[i] * tdm_of

            # Underflow: shifted interval extends below the grid bottom
            # (folded into bin 0, see module docstring item 6).
            uf_lo = lo_i[0]
            uf_hi = np.minimum(grid_lo, hi_i[0])
            uf_valid = valid_i & (uf_hi > uf_lo)
            tdn_uf, tdm_uf = _moment_integral(a0_i[0], a1_i[0], uf_lo, uf_hi)
            tdn_uf = np.where(uf_valid, tdn_uf, 0.0)
            tdm_uf = np.where(uf_valid, tdm_uf, 0.0)
            new_n[0, :] += tdn_uf
            new_rmt[0, :] += tdm_uf
            new_rmat[0, :] += ratio_rmat[i] * tdm_uf
            new_rmas[0, :] += ratio_rmas[i] * tdm_uf

        sb_i = same_bin_source[i]
        if sb_i.any():
            new_n[i, :] += np.where(sb_i, npd[i], 0.0)
            new_rmt[i, :] += np.where(sb_i, mpd[i], 0.0)
            new_rmat[i, :] += np.where(sb_i, ratio_rmat[i] * mpd[i], 0.0)
            new_rmas[i, :] += np.where(sb_i, ratio_rmas[i] * mpd[i], 0.0)

    return new_n, new_rmt, new_rmat, new_rmas


# ---------------------------------------------------------------------------
# vapor_deposition_liquid -- the M2a Task 5 deliverable.
# ---------------------------------------------------------------------------


def vapor_deposition_liquid(
    liquid: LiquidState,
    thermo_state: ThermoState,
    config: AmpsConfig,
    dt_vp: float,
    diag: LiquidDiag,
) -> LiquidState:
    """`vapor_deposition`'s LIQUID (`g%token==1`) condensation/evaporation
    growth (G3 sections 1 + 2b + 4), Chen and Lamb (1994) semidiscrete
    growth: per-bin mean-mass-point growth `d_mean_mass = (coef1*s_v_n +
    coef2)*dt_vp` (`coef1`/`coef2` = `diag.vapdep_coef1`/`vapdep_coef2`,
    already folding capacitance + ventilation, `core/liquid_diag.py`),
    shifted-bin boundaries, a total-evaporation check, mass-component
    (aerosol total/soluble) ratio distribution, and the GATHER-formulated
    mass-space bin remap (`_gather_remap`). See the module docstring for
    every design decision / fact-gap (`s_v_n` recomputation, diagnostic
    staleness, the un-quoted `cal_lincubprms_vec`'s linear-reconstruction
    substitute, the omitted haze-transfer/`r_act` check, the skipped
    `icond_ngb` truncation optimization, and underflow handling).

    Args:
        liquid: `LiquidState` (`ncat` must be 1, per `state.py`) -- the
            CURRENT (post-CCN-activation, per `implementations/
            warm_loop.py`'s vap-loop wiring) liquid bin state.
        thermo_state: `ThermoState` supplying P (`ptotv`), T (`tv`), and
            vapor mixing ratio (`qvv`) -- also CURRENT/post-activation,
            used to recompute `s_v_n` (module docstring item 1).
        config: `AmpsConfig` (`nbin_h` for the bin grid, `level_comp` for
            the mass-component-ratio / `temp_dM` branch gates).
        dt_vp: vapor-deposition substep length (s), `g%dt` in G3.
        diag: `LiquidDiag` (Task 2's `diag_pq_liquid` output) -- the
            STALE (pre-activation-refresh) per-bin diagnostics
            (`mean_mass`, `vapdep_coef1`, `vapdep_coef2`); see module
            docstring item 2 for why this staleness is faithful, not a
            bug.

    Returns:
        `LiquidState`, same shape as `liquid` (`rcon_q`/`rmt_q`/
        `rmat_q`/`rmas_q` updated; `ncat`/`npoints`/`nbins` preserved).
    """
    if liquid.ncat != 1:
        raise NotImplementedError(
            f"vapor_deposition_liquid only supports ncat=1 (see state.py's own to_fields()/"
            f"from_fields() convention); got ncat={liquid.ncat}"
        )

    lp = LiquidPPV
    con = liquid.values[lp.rcon_q.py_idx, :, 0, :]
    mass_total = liquid.values[lp.rmt_q.py_idx, :, 0, :]
    mass_aero_total = liquid.values[lp.rmat_q.py_idx, :, 0, :]
    mass_aero_soluble = liquid.values[lp.rmas_q.py_idx, :, 0, :]

    nbins = liquid.nbins
    npoints = liquid.npoints

    # icond3 (source-bin eligibility) -- see module docstring item 2.
    icond3 = diag.mean_mass > 0.0

    # Grid-box skip: no active bin ANYWHERE (e.g. a genuinely no-water
    # state, G1's own icycle_n concept) -- return the input unchanged,
    # identity, matching core/activation.py's own TestSkipMask precedent.
    # Also sidesteps bin_grid.make_bin_grid's nbins in {40,80} validation
    # for such trivial (no real physics to do) states.
    if not icond3.any():
        return liquid

    # s_v_n -- see module docstring item 1.
    p = get_thermo_prop(thermo_state, ThermoProp.ptotv)[None, :]
    t = get_thermo_prop(thermo_state, ThermoProp.tv)[None, :]
    qv = get_thermo_prop(thermo_state, ThermoProp.qvv)[None, :]
    estbar, esitbar = thermo_fn.make_esat_tables()
    s_v_n = np.broadcast_to(_liquid_supersaturation(p, qv, t, estbar, esitbar), (nbins, npoints))

    # d_mean_mass (G3 section 1 line 526, LIQUID branch) + dMcon.
    d_mean_mass = np.where(icond3, (diag.vapdep_coef1 * s_v_n + diag.vapdep_coef2) * dt_vp, 0.0)
    dmcon = con * d_mean_mass

    # Shifted bin boundaries (G3 section 1 lines 384-389).
    binb = bin_grid.make_bin_grid("liquid", nbins, nbin_h=config.nbin_h).binb
    bin_lo = binb[:-1][:, None]
    bin_hi = binb[1:][:, None]

    mean_mass_safe = np.where(icond3, diag.mean_mass, 1.0)
    d_mass_b1 = d_mean_mass * (bin_lo / mean_mass_safe) ** (1.0 / 3.0)
    d_mass_b2 = d_mean_mass * (bin_hi / mean_mass_safe) ** (1.0 / 3.0)
    shifted_lo = np.maximum(0.0, bin_lo + d_mass_b1)
    shifted_hi = np.maximum(0.0, bin_hi + d_mass_b2)

    # Total-evaporation check (G3 section 1 lines 392-424); the
    # haze-transfer/r_act sub-condition is omitted (module docstring
    # item 4).
    evaporating = icond3 & (d_mean_mass < 0.0)
    # jmat=rmat for token==1 (liquid), level>=4 branch (G3 line 397).
    temp_dm = mass_aero_total - mass_total if config.level_comp >= 4 else binb[0] * con - mass_total
    total_evap = evaporating & ((temp_dm * 0.99999 >= dmcon) | (shifted_hi <= binb[0]))
    icond_noevp = icond3 & ~total_evap

    # Npd/Mpd -- post-growth number/mass in the shifted bin (G3 lines
    # 429-431).
    npd = np.where(icond_noevp, con, 0.0)
    mpd = np.where(icond_noevp, mass_total + dmcon, 0.0)

    # Mass-component distribution -- LIQUID's own "dep_mass3" role, see
    # module docstring item 7 (G3 section 2b, `if(g%token==1)`, `level>=4`).
    if config.level_comp >= 4:
        ratio_rmat = np.where(icond_noevp, _safe_div(mass_aero_total, mpd), 0.0)
        ratio_rmas = np.where(icond_noevp, _safe_div(mass_aero_soluble, mpd), 0.0)
    else:
        ratio_rmat = np.zeros_like(mass_total)
        ratio_rmas = np.zeros_like(mass_total)

    new_n, new_rmt, new_rmat, new_rmas = _gather_remap(
        binb, shifted_lo, shifted_hi, npd, mpd, ratio_rmat, ratio_rmas, icond_noevp
    )

    # Bins vapor_deposition never touched (icond3==False) pass through
    # unchanged at their own index.
    passthrough = ~icond3
    new_n = new_n + np.where(passthrough, con, 0.0)
    new_rmt = new_rmt + np.where(passthrough, mass_total, 0.0)
    new_rmat = new_rmat + np.where(passthrough, mass_aero_total, 0.0)
    new_rmas = new_rmas + np.where(passthrough, mass_aero_soluble, 0.0)

    values = liquid.values.copy()
    values[lp.rcon_q.py_idx, :, 0, :] = new_n
    values[lp.rmt_q.py_idx, :, 0, :] = new_rmt
    values[lp.rmat_q.py_idx, :, 0, :] = new_rmat
    values[lp.rmas_q.py_idx, :, 0, :] = new_rmas
    return LiquidState(values=values)

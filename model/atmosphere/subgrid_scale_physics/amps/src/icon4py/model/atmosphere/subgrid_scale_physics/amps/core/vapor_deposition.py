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
(section 2), the mass-space bin remap `cal_transbin_vec`/`shift_bin_loop1`
(section 4, `mod_amps_utility.F90:8713`), and the aerosol/vapor return
paths (`mod_amps_core.F90` lines 616-659 "Aerosol-evaporation gather",
`cal_transbin_vec`'s own `iaer_src==1` underflow branch, and
`add_tendency_ap_vec`, `mod_amps_core.F90:21606-21665`, read directly per
this task's dispatch authorization -- none of the three are quoted in
full by G3). Also directly read (dispatch-authorized, not quoted by G3):
`cal_lincubprms_vec` (`mod_amps_utility.F90:9798-10031`) -- see item 3
below; this REPLACES an earlier draft of this module that incorrectly
believed the routine was unavailable. Ice-only work (habit/growth-mode,
`assign_Qp_v3_vec`, `cal_xxx_p_v5_vec`, axis growth, `dep_mass3_vec2`'s
ice-mass-component partition -- M3 scope) is NOT ported here.

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
matching G3's per-routine liquid/ice split table. The SAME reasoning
applies to the "Aerosol-evaporation gather"/`cal_transbin_vec`
`iaer_src==1`/`add_tendency_ap_vec` code paths added in this revision:
none of their own bodies are gated on `g%token` at all (only `jmat`,
already resolved to `rmat` for token==1 upstream, appears), so they are
equally reachable for liquid.

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
3. **`cal_lincubprms_vec` IS ported here (`mod_amps_utility.F90:
   9798-10031`, read directly, dispatch-authorized -- an earlier draft
   of this module incorrectly treated it as unavailable/unquoted by G3
   and substituted a same-bin passthrough for ALL negative-density cases;
   that substitution is now REMOVED except as this module's own
   last-resort tail fallback, see below). The real routine, per bin:
     (a) a base 2-moment linear fit (`n(x)=a1+a3*(x-a2)`, `a2`=midpoint)
         -- mathematically identical to a naive "solve 2 moment
         equations for a linear density" (this WAS correctly derived
         independently in the earlier draft; the Fortran's own
         parametrization is just a shifted-coordinate rewrite of the
         SAME unique solution -- verified algebraically, see
         `_linear_fit`'s own docstring);
     (b) if that fit's density goes negative at either edge, a
         TRUNCATED-SUPPORT linear re-fit vanishing exactly at a
         recomputed point `x0` (`_linear_fit`'s `neg_left`/`neg_right`
         branches) -- NOT a same-bin passthrough: the support shrinks to
         `[x0,bd2]` or `[bd1,x0]` and STILL participates in the
         multi-bin gather over that narrower interval, still exactly
         N/M-conserving (verified algebraically: `_linear_fit`'s own
         docstring);
     (c) for INTERIOR bins (not first/last) whose own fit AND both
         immediate neighbors' fits are all valid (`error==0`, (a) or
         (b) above), a Legendre-polynomial-basis CUBIC upgrade informed
         by the neighbors' own reconstructed density at their mean-mass
         point (Dinh and Durran 2012) -- `_cubic_upgrade`, used
         whenever it passes its own validity checks (no interior
         negative-density extremum, non-negative at both edges, no
         numerical blow-up), else the bin keeps (a)/(b)'s result. THIS,
         not the simple linear fit, is the common case for well-behaved
         interior bins.
   **A literal-Fortran-fidelity finding**: `_cubic_upgrade`'s `b3`
   coefficient (`mod_amps_utility.F90:9942`,
   `b3=(C_L*A_R-A_L*C_R)/B_L*A_R-A_L*B_R`) parses, per Fortran's
   left-to-right `*`/`/` associativity, as `((C_L*A_R-A_L*C_R)/B_L)*A_R
   - A_L*B_R` -- NOT the Cramer's-rule `(C_L*A_R-A_L*C_R)/(B_L*A_R-A_L*
   B_R)` that a commented-out alternate two lines below it
   (`!b3=(...)/max(1e-150,B_L*A_R-A_L*B_R)`) suggests was actually
   intended. This module ports the LITERAL (active, compiled) formula,
   per this task's Fortran-fidelity mandate -- flagged prominently here
   and in `_cubic_upgrade`'s own docstring; NOT "fixed" to the
   presumably-intended Cramer's-rule form.
   **What is still NOT reproduced** (a narrower tail than the earlier
   draft's blanket simplification): `vapor_deposition`'s OWN further
   escalation when `cal_lincubprms_vec` itself reports a genuinely
   unusable bin (`error_number` outside `{0}` after (a)/(b)/(c) above --
   i.e. `_reconstruct_distribution`'s `ok=False`) -- the real Fortran
   retries with `bbmf`-widened bin boundaries and `cal_linprms_vec_s`
   (G3 section 1 lines 546-592); this module instead falls back to a
   same-bin (no shift) passthrough for such bins, still exactly
   N/M-conserving, not reproducing that further multi-stage retry. This
   tail is now genuinely rare (only bins where even the truncated-support
   fit is itself geometrically invalid, e.g. the recomputed vanishing
   point falls outside the bin), not the common negative-density case
   (which (b) above now handles faithfully).
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
   outcome; this module's reconstruction/gather-remap handles arbitrarily
   small shifts exactly (conserving N/M to machine precision) without
   any special-casing, so omitting the optimization changes no result,
   only simplifies this port.
6. **Total evaporation and underflow now correctly divert to
   `AerosolState`/vapor (`ThermoState.qvv`), not silently discarded.**
   Ported from THREE Fortran sources, none quoted by G3, all read
   directly (dispatch-authorized): the "Aerosol-evaporation gather"
   (`mod_amps_core.F90` lines 616-659, total-evaporation bins:
   number->`ap_dN`, aerosol total/soluble mass->`ap_dM`/`ap_dMS`,
   water-only remainder->vapor via `used_v`), `cal_transbin_vec`'s own
   `iaer_src==1` underflow branch (`mod_amps_utility.F90` lines
   1736-1777 in G3's own quoted excerpt -- underflow PIECES, not whole
   bins, divert their aerosol-weighted share the same way), and
   `add_tendency_ap_vec` (`mod_amps_core.F90:21606-21665` -- ALWAYS
   returns to `ga(1)`/`ga(2)` i.e. `AerosolState` category 0/1
   REGARDLESS of the aerosol's original category, keyed on
   `eps_map>=lmt_frac` (~1e-5): mostly-soluble -> category 0,
   mostly-insoluble -> category 1). `AerosolState`'s own single-bin
   (`nbins=1`, `MS(1,n)`) per-category convention is
   `core/activation.py`'s own established one (its own docstring: "bin
   INDEX 0 for each category"), reused here, NOT reinvented.

   **One deliberate, documented DEVIATION from the literal underflow
   branch**: `cal_transbin_vec`'s own `ap_dMV(n,icat)=ap_dMV(n,icat)+
   trans_dM` credits the FULL underflow-piece mass (water AND aerosol
   together) to vapor via `add_tendency_ap_vec`'s
   `ag%TV(n)%dmassdt_v(icat)+=ap_dMV(n,icat)/dt` -- but that SAME
   piece's aerosol-weighted share ALSO gets returned to the aerosol
   group's own mass (`ap_dM`/`ap_dMS`). Read literally, this DOUBLE-
   COUNTS the aerosol mass (once as returned dry aerosol, once as
   vaporized water) -- inconsistent with the "Aerosol-evaporation
   gather" section's OWN unambiguous water-ONLY vapor formula
   (`max(0,mass(1)-mass(jmat))`, i.e. total-minus-aerosol) for the
   whole-bin total-evaporation case. Given this task's explicit,
   paramount test requirement (total water AND total aerosol mass/number
   conserved to 1e-12), this module credits vapor with the underflow
   piece's WATER-ONLY remainder (`trans_dM - aerosol_share`) instead of
   the literal (apparently double-counting) `trans_dM` -- the physically
   consistent, conservation-preserving choice, matching the whole-bin
   case's own unambiguous formula. Flagged here and in
   `_underflow_pieces`'s own docstring as a genuine, reasoned deviation,
   not a silent one.

   `ncat<2` (this warm-only port's own toy/test `AerosolState` fixtures
   commonly carry a single category): both `add_tendency_ap_vec`
   destination categories collapse into category 0 -- a documented
   simplification, matching Task 4's own "no dust category" precedent.
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
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import (
    AerosolPPV,
    LiquidPPV,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


# lmt_frac -- soluble-fraction threshold selecting which of the 2 returned
# aerosol categories a fully/partially-evaporated bin's content goes to
# (G3 section 1 line 216, `add_tendency_ap_vec`'s own callers).
LMT_FRAC = 0.99999e-5


# ---------------------------------------------------------------------------
# Small numeric helpers -- see core/liquid_diag.py's/core/activation.py's own
# identical-purpose, independently-duplicated `_safe_div` precedent.
# ---------------------------------------------------------------------------


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )


def _fsign1(x: np.ndarray) -> np.ndarray:
    """Fortran two-argument `SIGN(1.0,x)`: `+1` if `x>=0`, `-1` if `x<0`
    (NOT `numpy.sign`, which returns `0` at `x==0` -- Fortran's `SIGN`
    never does)."""
    return np.where(x >= 0.0, 1.0, -1.0)


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
# _moment_integral -- exact integral of the general cubic
# n(x)=a0+a1*x+a2*x^2+a3*x^3 over [bd1,bd2] -- cal_transbin_vec's own
# shift_bin_loop1 formula (G3 section 4), quoted FULL.
# ---------------------------------------------------------------------------


def _moment_integral(  # noqa: PLR0917 [too-many-positional-arguments]
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, bd1: np.ndarray, bd2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact integral of `n(x)=a0+a1*x+a2*x^2+a3*x^3` over `[bd1,bd2]`:
    0th moment (`trans_dN`) and 1st moment (`trans_dM`) --
    `cal_transbin_vec`'s own `shift_bin_loop1` formula (G3 section 4,
    quoted FULL, lines 8899-8919/1672-1684), verbatim (including its own
    `max(0,...)` clamp on both outputs)."""
    x_a = bd2 + bd1
    x_b = bd2 - bd1
    x_c = bd2 * bd1
    x_d = bd2 * bd2 + bd1 * bd1
    trans_dn = (
        a0 * x_b + 0.5 * a1 * x_a * x_b + a2 * x_b * (x_d + x_c) / 3.0 + 0.25 * a3 * x_d * x_a * x_b
    )
    trans_dm = (
        0.5 * a0 * x_a * x_b
        + a1 * x_b * (x_d + x_c) / 3.0
        + 0.25 * a2 * x_d * x_a * x_b
        + 0.2 * a3 * x_b * (x_a * x_a * (x_d - x_c) + x_c * x_c)
    )
    return np.maximum(trans_dn, 0.0), np.maximum(trans_dm, 0.0)


# ---------------------------------------------------------------------------
# cal_lincubprms_vec (mod_amps_utility.F90:9798-10031) -- see module
# docstring item 3.
# ---------------------------------------------------------------------------


def _linear_fit(
    number: np.ndarray, mass: np.ndarray, bd1: np.ndarray, bd2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`cal_lincubprms_vec`'s first+second do-loops (lines 9832-9892): the
    base 2-moment linear fit `n(x)=a1+a3*(x-a2)` (`a2`=bin midpoint --
    algebraically the SAME unique linear density matching the given
    0th/1st moments over `[bd1,bd2]` a naive from-scratch 2-moment solve
    gives, just parametrized around the midpoint instead of the origin;
    verified by direct algebraic substitution), THEN the negative-density
    truncated-support re-fit when that fit's density goes negative at
    either edge: `a1` becomes a SENTINEL (`-1.0`: density negative at
    `bd1`, re-fit vanishes at a recomputed `x0=3*mean_mass-2*bd2` with
    support `[x0,bd2]`; `-2.0`: density negative at `bd2`, re-fit
    vanishes at `x0=3*mean_mass-2*bd1` with support `[bd1,x0]`) -- `a2`
    becomes that `x0`, `a3` the re-fit's own slope. Returns `(a1, a2, a3,
    ok)` in this dual representation (`n(x)=max(0,a1)+a3*(x-a2)`
    uniformly, per the Fortran's own later consumers -- see
    `_reconstruct_distribution`); `ok=False` collapses the Fortran's own
    `error_number` codes (0 vs 1/2/3/4/10) into a single "usable" flag
    (nothing downstream of this module consumes the specific integer
    code, only whether it is zero)."""
    degenerate = ((bd1 < 1.0e-30) & (bd2 < 1.0e-30)) | (bd1 == bd2)
    valid_moments = (number > 0.0) & (mass > 0.0)
    ok0 = ~degenerate & valid_moments

    width = bd2 - bd1
    safe_width = np.where(ok0, width, 1.0)
    x0_mid = 0.5 * (bd1 + bd2)
    a1 = np.where(ok0, number / safe_width, 0.0)
    a2 = np.where(ok0, x0_mid, 0.0)
    a3 = np.where(ok0, 12.0 * (mass - x0_mid * number) / safe_width**3, 0.0)

    density_bd1 = a1 + a3 * (bd1 - a2)
    density_bd2 = a1 + a3 * (bd2 - a2)
    neg_left = ok0 & (density_bd1 < 0.0)
    neg_right = ok0 & ~neg_left & (density_bd2 < 0.0)

    mean_mass = _safe_div(mass, number)

    x0_left = 3.0 * mean_mass - 2.0 * bd2
    a3_left = 2.0 * number / np.maximum((bd2 - x0_left) ** 2, 1.0e-100)
    left_valid = bd2 >= x0_left  # error==0 sub-case (G3: floor(0.5*(1-sign(1,bd2-x0)))==0)

    x0_right = 3.0 * mean_mass - 2.0 * bd1
    a3_right = -2.0 * number / np.maximum((bd1 - x0_right) ** 2, 1.0e-100)
    right_valid = x0_right >= 0.0  # error==0 sub-case (G3: floor(1-sign(1,x0))==0)

    a1 = np.where(neg_left, -1.0, np.where(neg_right, -2.0, a1))
    a2 = np.where(neg_left, x0_left, np.where(neg_right, x0_right, a2))
    a3 = np.where(neg_left, a3_left, np.where(neg_right, a3_right, a3))

    ok = ok0 & np.where(neg_left, left_valid, np.where(neg_right, right_valid, True))

    blown_up = (np.abs(a1) > 1.0e100) | (np.abs(a2) > 1.0e100) | (np.abs(a3) > 1.0e100)
    ok = ok & ~blown_up & np.isfinite(a1) & np.isfinite(a2) & np.isfinite(a3)

    a1 = np.where(ok, a1, 0.0)
    a2 = np.where(ok, a2, 0.0)
    a3 = np.where(ok, a3, 0.0)
    return a1, a2, a3, ok


def _cubic_upgrade(  # noqa: PLR0917 [too-many-statements/positional-arguments]
    number: np.ndarray,
    mass: np.ndarray,
    bd1: np.ndarray,
    bd2: np.ndarray,
    a1_lin: np.ndarray,
    a2_lin: np.ndarray,
    a3_lin: np.ndarray,
    ok_lin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`cal_lincubprms_vec`'s "second, calculate the cubic" block (lines
    9908-10005): a Legendre-polynomial-basis cubic reconstruction for
    INTERIOR bins (0-based `1<=i<=nbins-2`) whose own linear fit AND both
    immediate neighbors' linear fits are valid (`ok_lin`), informed by
    each neighbor's own reconstructed density AT ITS OWN mean-mass point
    (Dinh and Durran 2012). All inputs/outputs `(nbins, npoints)`.

    See module docstring item 3 for the `b3` literal-vs-intended
    operator-precedence finding -- ported LITERALLY
    (`b3=((C_L*A_R-A_L*C_R)/B_L)*A_R - A_L*B_R`, matching Fortran's own
    left-to-right `*`/`/` associativity on the ACTIVE line, not the
    commented-out Cramer's-rule alternate).

    Returns `(a0_c, a1_c, a2_c, a3_c, i_cubic)`: standard polynomial
    coefficients over the FULL (non-truncated) `[bd1,bd2]`, and the
    validity mask (Fortran's own `i_cubic`, PLUS an added `np.isfinite`
    guard beyond the literal Fortran -- a defensive addition, not itself
    part of the ported algorithm, preventing NaN/Inf from ever reaching
    the final reconstruction when `b3`'s divisions are near-singular)."""
    nbins = number.shape[0]
    interior = np.zeros_like(ok_lin, dtype=bool)
    interior[1:-1, :] = True

    idx = np.arange(nbins)
    left = np.roll(idx, 1)
    right = np.roll(idx, -1)

    number_l, number_r = number[left], number[right]
    mass_l, mass_r = mass[left], mass[right]
    a1_l, a2_l, a3_l = a1_lin[left], a2_lin[left], a3_lin[left]
    a1_r, a2_r, a3_r = a1_lin[right], a2_lin[right], a3_lin[right]
    ok_l, ok_r = ok_lin[left], ok_lin[right]

    eligible = interior & ok_lin & ok_l & ok_r

    delta_m = bd2 - bd1
    am0 = 0.5 * (bd1 + bd2)
    safe_delta_m = np.where(eligible, delta_m, 1.0)

    b0 = np.where(eligible, number / safe_delta_m, 0.0)
    b1 = np.where(eligible, 6.0 * (mass - am0 * number) / safe_delta_m**2, 0.0)

    i_con_gt0_left = number_l > 1.0e-30
    i_con_gt0_right = number_r > 1.0e-30

    amb_l = _safe_div(mass_l, number_l)
    anb_l = np.maximum(0.0, a1_l) + a3_l * (amb_l - a2_l)
    xib_l = 2.0 * (amb_l - am0) / safe_delta_m

    amb_r = _safe_div(mass_r, number_r)
    anb_r = np.maximum(0.0, a1_r) + a3_r * (amb_r - a2_r)
    xib_r = 2.0 * (amb_r - am0) / safe_delta_m

    a_l = 0.5 * (3.0 * xib_l * xib_l - 1.0)
    b_l = 0.5 * (5.0 * xib_l**3 - 3.0 * xib_l)
    c_l = anb_l - b0 - b1 * xib_l

    a_r = 0.5 * (3.0 * xib_r * xib_r - 1.0)
    b_r = 0.5 * (5.0 * xib_r**3 - 3.0 * xib_r)
    c_r = anb_r - b0 - b1 * xib_r

    with np.errstate(divide="ignore", invalid="ignore"):
        # LITERAL Fortran precedence -- see this function's own docstring.
        b3 = (c_l * a_r - a_l * c_r) / b_l * a_r - a_l * b_r
        b2 = (c_l - b3 * b_l) / a_l

        q = am0 / safe_delta_m
        a0_c = b0 - 2.0 * q * b1 + (6.0 * q * q - 0.5) * b2 - (20.0 * q**3 - 3.0 * q) * b3
        a1_c = (2.0 * b1 - 12.0 * q * b2 + (60.0 * q * q - 3.0) * b3) / safe_delta_m
        a2_c = (6.0 * b2 - 60.0 * q * b3) / safe_delta_m**2
        a3_c = 20.0 * b3 / safe_delta_m**3

        delta = a2_c * a2_c - 3.0 * a1_c * a3_c
        am_e = (-a2_c + np.sqrt(np.maximum(0.0, delta))) / (3.0 * a3_c)
        an_me = a0_c + a1_c * am_e + a2_c * am_e * am_e + a3_c * am_e**3

    i_delta_ge0 = delta >= 0.0
    i_ame_lt0 = an_me < 0.0

    an_ml = a0_c + a1_c * bd1 + a2_c * bd1 * bd1 + a3_c * bd1**3
    an_mr = a0_c + a1_c * bd2 + a2_c * bd2 * bd2 + a3_c * bd2**3
    i_anml_lt0 = an_ml < 0.0
    i_anmr_lt0 = an_mr < 0.0

    i_large = np.abs(a0_c) > 1.0e30
    finite = np.isfinite(a0_c) & np.isfinite(a1_c) & np.isfinite(a2_c) & np.isfinite(a3_c)

    i_cubic = (
        eligible
        & i_con_gt0_left
        & i_con_gt0_right
        & ~(i_delta_ge0 & i_ame_lt0)
        & ~i_anml_lt0
        & ~i_anmr_lt0
        & ~i_large
        & finite
    )

    a0_c = np.where(i_cubic, a0_c, 0.0)
    a1_c = np.where(i_cubic, a1_c, 0.0)
    a2_c = np.where(i_cubic, a2_c, 0.0)
    a3_c = np.where(i_cubic, a3_c, 0.0)
    return a0_c, a1_c, a2_c, a3_c, i_cubic


def _reconstruct_distribution(
    number: np.ndarray, mass: np.ndarray, bd1: np.ndarray, bd2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full `cal_lincubprms_vec`: `_linear_fit` (base + truncated-support
    fallback) + `_cubic_upgrade` (interior Legendre reconstruction).
    Returns `(poly_a0, poly_a1, poly_a2, poly_a3, eff_bd1, eff_bd2, ok)`
    ready for `_moment_integral`:
    `n(x)=poly_a0+poly_a1*x+poly_a2*x^2+poly_a3*x^3` over `[eff_bd1,
    eff_bd2]` (a SUBSET of `[bd1,bd2]` only for truncated-support linear
    bins; unchanged for cubic-upgraded and plain-linear bins). `ok=False`:
    genuinely unusable (this module's OWN further, narrower fallback --
    a same-bin passthrough -- handles those, see module docstring item
    3's final paragraph)."""
    a1_lin, a2_lin, a3_lin, ok = _linear_fit(number, mass, bd1, bd2)

    truncated_left = ok & (a1_lin == -1.0)
    truncated_right = ok & (a1_lin == -2.0)
    eff_bd1 = np.where(truncated_right, bd1, np.where(truncated_left, a2_lin, bd1))
    eff_bd2 = np.where(truncated_left, bd2, np.where(truncated_right, a2_lin, bd2))

    intercept = np.where(a1_lin >= 0.0, a1_lin, 0.0)  # max(0,a1)
    poly_a0 = np.where(ok, intercept - a3_lin * a2_lin, 0.0)
    poly_a1 = np.where(ok, a3_lin, 0.0)
    poly_a2 = np.zeros_like(poly_a0)
    poly_a3 = np.zeros_like(poly_a0)

    a0_c, a1_c, a2_c, a3_c, i_cubic = _cubic_upgrade(
        number, mass, bd1, bd2, a1_lin, a2_lin, a3_lin, ok
    )
    poly_a0 = np.where(i_cubic, a0_c, poly_a0)
    poly_a1 = np.where(i_cubic, a1_c, poly_a1)
    poly_a2 = np.where(i_cubic, a2_c, poly_a2)
    poly_a3 = np.where(i_cubic, a3_c, poly_a3)
    eff_bd1 = np.where(i_cubic, bd1, eff_bd1)
    eff_bd2 = np.where(i_cubic, bd2, eff_bd2)

    # Defensive (the Fortran's own error-code design already excludes
    # this via `ok`, see _linear_fit's left_valid/right_valid): a
    # genuinely inverted/zero-width truncated support is not usable.
    ok = ok & (eff_bd2 > eff_bd1)
    poly_a0 = np.where(ok, poly_a0, 0.0)
    poly_a1 = np.where(ok, poly_a1, 0.0)
    poly_a2 = np.where(ok, poly_a2, 0.0)
    poly_a3 = np.where(ok, poly_a3, 0.0)

    return poly_a0, poly_a1, poly_a2, poly_a3, eff_bd1, eff_bd2, ok


# ---------------------------------------------------------------------------
# cal_transbin_vec / shift_bin_loop1 (G3 section 4, iphase==1) -- GATHER
# formulation: for each destination bin, sum contributions from every
# source bin whose (possibly truncated/cubic-reconstructed) shifted
# interval overlaps it (the dense i x ibx double loop G3 quotes verbatim),
# vectorized per source bin over (destination bin, point) -- matching
# core/activation.py's own `_add_simple` per-source-bin-loop convention.
# ---------------------------------------------------------------------------


def _gather_remap(  # noqa: PLR0917 [too-many-positional-arguments]
    binb: np.ndarray,
    poly_a0: np.ndarray,
    poly_a1: np.ndarray,
    poly_a2: np.ndarray,
    poly_a3: np.ndarray,
    eff_bd1: np.ndarray,
    eff_bd2: np.ndarray,
    ratio_rmat: np.ndarray,
    ratio_rmas: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GATHER mass-space bin remap, LIQUID-GRID portion only (the part of
    each source bin's reconstructed distribution that overlaps
    `[binb[0],binb[-1]]`) -- underflow (below `binb[0]`) is handled
    SEPARATELY by `_underflow_pieces` (diverted to aerosol/vapor, module
    docstring item 6), NOT folded in here.

    `binb`: `(nbins+1,)` bin boundaries. `poly_a0..a3`/`eff_bd1`/
    `eff_bd2`/`ratio_rmat`/`ratio_rmas`/`source_valid`: `(nbins,
    npoints)`, one entry per SOURCE bin (`_reconstruct_distribution`'s
    own output, plus the mass-component ratios and `icond_noevp`
    eligibility mask).

    Overflow (`eff_bd2>binb[-1]`) folds into the LAST destination bin
    (G3's own `cal_transbin_vec` overflow branch, additive with the
    grid-clipped contribution -- the two pieces are disjoint, meeting
    exactly at `binb[-1]`, never double-counted).

    Returns `(new_n, new_rmt, new_rmat, new_rmas)`, each `(nbins,
    npoints)`.
    """
    nbins = binb.shape[0] - 1
    npoints = eff_bd1.shape[1]
    new_n = np.zeros((nbins, npoints))
    new_rmt = np.zeros((nbins, npoints))
    new_rmat = np.zeros((nbins, npoints))
    new_rmas = np.zeros((nbins, npoints))

    bin_lo = binb[:-1][:, None]
    bin_hi = binb[1:][:, None]
    grid_lo = binb[0]
    grid_hi = binb[-1]

    for i in range(eff_bd1.shape[0]):
        valid_i = source_valid[i]
        if not valid_i.any():
            continue
        lo_i = eff_bd1[i][None, :]
        hi_i = eff_bd2[i][None, :]
        a0_i = poly_a0[i][None, :]
        a1_i = poly_a1[i][None, :]
        a2_i = poly_a2[i][None, :]
        a3_i = poly_a3[i][None, :]

        # Main grid-clipped overlap, dense over every destination bin
        # (the i x ibx loop, G3 section 4 lines 8899-8919/1656-1665).
        ov_lo = np.maximum(bin_lo, np.maximum(lo_i, grid_lo))
        ov_hi = np.minimum(bin_hi, np.minimum(hi_i, grid_hi))
        overlap = (ov_hi > ov_lo) & valid_i[None, :]
        bd1 = np.where(overlap, ov_lo, 0.0)
        bd2 = np.where(overlap, ov_hi, bd1)
        tdn, tdm = _moment_integral(a0_i, a1_i, a2_i, a3_i, bd1, bd2)
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
        tdn_of, tdm_of = _moment_integral(a0_i[0], a1_i[0], a2_i[0], a3_i[0], of_lo, of_hi)
        tdn_of = np.where(of_valid, tdn_of, 0.0)
        tdm_of = np.where(of_valid, tdm_of, 0.0)
        new_n[-1, :] += tdn_of
        new_rmt[-1, :] += tdm_of
        new_rmat[-1, :] += ratio_rmat[i] * tdm_of
        new_rmas[-1, :] += ratio_rmas[i] * tdm_of

    return new_n, new_rmt, new_rmat, new_rmas


def _underflow_pieces(  # noqa: PLR0917 [too-many-positional-arguments]
    binb: np.ndarray,
    poly_a0: np.ndarray,
    poly_a1: np.ndarray,
    poly_a2: np.ndarray,
    poly_a3: np.ndarray,
    eff_bd1: np.ndarray,
    eff_bd2: np.ndarray,
    ratio_rmat: np.ndarray,
    ratio_rmas: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The portion of each source bin's reconstructed distribution BELOW
    `binb[0]` -- `cal_transbin_vec`'s own `iaer_src==1` underflow branch
    (G3 section 4, quoted FULL, lines 1736-1777), diverted to the
    aerosol group / vapor by the caller (`vapor_deposition_liquid`), NOT
    folded back into liquid bin 0 (module docstring item 6 -- this
    REPLACES an earlier draft's "fold into bin 0" simplification, which
    silently discarded the returned aerosol/vapor mass).

    Returns, each `(nbins, npoints)` (one entry per SOURCE bin, NOT yet
    summed into aerosol categories -- the caller needs the per-bin
    `eps_map` to categorize):
      `piece_number`: 0th moment of the underflow piece.
      `piece_aero_mass`: `ratio_rmat[i]*piece_total_mass` (aerosol
        total-mass share).
      `piece_aero_soluble`: `ratio_rmas[i]*piece_total_mass` (aerosol
        soluble-mass share).
      `piece_water_mass`: `piece_total_mass - piece_aero_mass` (the
        WATER-only remainder -- see module docstring item 6's own
        deviation note: NOT the literal Fortran's full-mass `ap_dMV`).
    """
    grid_lo = binb[0]
    piece_number = np.zeros_like(eff_bd1)
    piece_aero_mass = np.zeros_like(eff_bd1)
    piece_aero_soluble = np.zeros_like(eff_bd1)
    piece_water_mass = np.zeros_like(eff_bd1)

    for i in range(eff_bd1.shape[0]):
        valid_i = source_valid[i] & (eff_bd1[i] < grid_lo)
        if not valid_i.any():
            continue
        uf_lo = eff_bd1[i]
        uf_hi = np.minimum(grid_lo, eff_bd2[i])
        uf_valid = valid_i & (uf_hi > uf_lo)
        tdn, tdm = _moment_integral(poly_a0[i], poly_a1[i], poly_a2[i], poly_a3[i], uf_lo, uf_hi)
        tdn = np.where(uf_valid, tdn, 0.0)
        tdm = np.where(uf_valid, tdm, 0.0)
        aero_mass = ratio_rmat[i] * tdm
        aero_soluble = np.minimum(ratio_rmas[i] * tdm, aero_mass)
        piece_number[i] = tdn
        piece_aero_mass[i] = aero_mass
        piece_aero_soluble[i] = aero_soluble
        piece_water_mass[i] = np.maximum(0.0, tdm - aero_mass)

    return piece_number, piece_aero_mass, piece_aero_soluble, piece_water_mass


# ---------------------------------------------------------------------------
# Aerosol-return category selection -- add_tendency_ap_vec
# (mod_amps_core.F90:21606-21665), see module docstring item 6.
# ---------------------------------------------------------------------------


def _evap_category_index(eps_map: np.ndarray, ncat: int) -> np.ndarray:
    """`eps_map>=LMT_FRAC` (essentially fully soluble) -> category 0;
    else category 1 -- ALWAYS one of these two, regardless of the
    aerosol's original category (`add_tendency_ap_vec`'s own hardcoded
    `ga(1)`/`ga(2)`). Clamped to `ncat-1` when `ncat<2` (module docstring
    item 6)."""
    cat = np.where(eps_map >= LMT_FRAC, 0, 1)
    return np.minimum(cat, max(ncat - 1, 0))


def _scatter_add_by_category(
    target: np.ndarray, values: np.ndarray, cat_index: np.ndarray, ncat: int
) -> np.ndarray:
    """Add `values` (`(npoints,)`) into `target` (`(ncat, npoints)`) at
    the per-point category given by `cat_index` (`(npoints,)`, values in
    `[0,ncat)`). Looped over the small, FIXED category count (<=4 for
    this codebase's own `AmpsConfig`), matching this module's own
    established per-source-bin-loop / GATHER convention -- no data-
    dependent scatter."""
    out = target.copy()
    for c in range(ncat):
        out[c, :] += np.where(cat_index == c, values, 0.0)
    return out


# ---------------------------------------------------------------------------
# vapor_deposition_liquid -- the M2a Task 5 deliverable.
# ---------------------------------------------------------------------------


def vapor_deposition_liquid(  # noqa: PLR0915, PLR0917 [too-many-statements/positional-arguments]
    liquid: LiquidState,
    aerosol: AerosolState,
    thermo_state: ThermoState,
    config: AmpsConfig,
    dt_vp: float,
    diag: LiquidDiag,
) -> tuple[LiquidState, AerosolState, ThermoState]:
    """`vapor_deposition`'s LIQUID (`g%token==1`) condensation/evaporation
    growth (G3 sections 1 + 2b + 4, PLUS the aerosol/vapor-return code
    paths listed in the module docstring item 6), Chen and Lamb (1994)
    semidiscrete growth: per-bin mean-mass-point growth `d_mean_mass =
    (coef1*s_v_n + coef2)*dt_vp` (`coef1`/`coef2` =
    `diag.vapdep_coef1`/`vapdep_coef2`, already folding capacitance +
    ventilation, `core/liquid_diag.py`), shifted-bin boundaries, a
    total-evaporation check, mass-component (aerosol total/soluble) ratio
    distribution, the full `cal_lincubprms_vec` reconstruction (linear +
    truncated-support + cubic, `_reconstruct_distribution`) and its
    GATHER-formulated mass-space bin remap (`_gather_remap`), and
    aerosol/vapor diversion for total-evaporation bins and underflow
    pieces (`_underflow_pieces`). See the module docstring for every
    design decision / fact-gap.

    Args:
        liquid: `LiquidState` (`ncat` must be 1, per `state.py`) -- the
            CURRENT (post-CCN-activation, per `implementations/
            warm_loop.py`'s vap-loop wiring) liquid bin state.
        aerosol: `AerosolState` -- the CURRENT aerosol group state
            (category 0/1 receive returned mass/number on evaporation,
            module docstring item 6).
        thermo_state: `ThermoState` supplying P (`ptotv`), T (`tv`), and
            vapor mixing ratio (`qvv`) -- also CURRENT/post-activation,
            used to recompute `s_v_n` (module docstring item 1) and
            updated on return with the net vapor exchanged this substep.
        config: `AmpsConfig` (`nbin_h` for the bin grid, `level_comp` for
            the mass-component-ratio / `temp_dM` branch gates).
        dt_vp: vapor-deposition substep length (s), `g%dt` in G3.
        diag: `LiquidDiag` (Task 2's `diag_pq_liquid` output) -- the
            STALE (pre-activation-refresh) per-bin diagnostics
            (`mean_mass`, `vapdep_coef1`, `vapdep_coef2`); see module
            docstring item 2 for why this staleness is faithful, not a
            bug.

    Returns:
        `(liquid', aerosol', thermo')`: `liquid'` has `rcon_q`/`rmt_q`/
        `rmat_q`/`rmas_q` updated (same shape as `liquid`); `aerosol'`
        gains the number/mass diverted from total-evaporation bins and
        underflow pieces; `thermo'`'s `qvv` gains the net water mass
        exchanged between vapor and liquid this substep (condensation
        consumes vapor, evaporation releases it, INCLUDING the
        underflow/total-evaporation water-only remainder).
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
        return liquid, aerosol, thermo_state

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

    poly_a0, poly_a1, poly_a2, poly_a3, eff_bd1, eff_bd2, ok = _reconstruct_distribution(
        npd, mpd, shifted_lo, shifted_hi
    )
    gather_source = icond_noevp & ok
    same_bin_source = icond_noevp & ~ok  # this module's own further fallback, see item 3

    new_n, new_rmt, new_rmat, new_rmas = _gather_remap(
        binb,
        poly_a0,
        poly_a1,
        poly_a2,
        poly_a3,
        eff_bd1,
        eff_bd2,
        ratio_rmat,
        ratio_rmas,
        gather_source,
    )
    uf_number, uf_aero_mass, uf_aero_soluble, uf_water_mass = _underflow_pieces(
        binb,
        poly_a0,
        poly_a1,
        poly_a2,
        poly_a3,
        eff_bd1,
        eff_bd2,
        ratio_rmat,
        ratio_rmas,
        gather_source,
    )

    # Same-bin (no shift) passthrough for genuinely-unusable reconstructions.
    new_n = new_n + np.where(same_bin_source, npd, 0.0)
    new_rmt = new_rmt + np.where(same_bin_source, mpd, 0.0)
    new_rmat = new_rmat + np.where(same_bin_source, ratio_rmat * mpd, 0.0)
    new_rmas = new_rmas + np.where(same_bin_source, ratio_rmas * mpd, 0.0)

    # Bins vapor_deposition never touched (icond3==False) pass through
    # unchanged at their own index.
    passthrough = ~icond3
    new_n = new_n + np.where(passthrough, con, 0.0)
    new_rmt = new_rmt + np.where(passthrough, mass_total, 0.0)
    new_rmat = new_rmat + np.where(passthrough, mass_aero_total, 0.0)
    new_rmas = new_rmas + np.where(passthrough, mass_aero_soluble, 0.0)

    liquid_values = liquid.values.copy()
    liquid_values[lp.rcon_q.py_idx, :, 0, :] = new_n
    liquid_values[lp.rmt_q.py_idx, :, 0, :] = new_rmt
    liquid_values[lp.rmat_q.py_idx, :, 0, :] = new_rmat
    liquid_values[lp.rmas_q.py_idx, :, 0, :] = new_rmas
    liquid_out = LiquidState(values=liquid_values)

    # ---- Aerosol/vapor diversion (module docstring item 6) -----------------

    ncat = aerosol.ncat
    ap = AerosolPPV
    aerosol_values = aerosol.values.copy()
    amt = aerosol_values[ap.amt_q.py_idx, 0, :, :]
    acon = aerosol_values[ap.acon_q.py_idx, 0, :, :]
    ams = aerosol_values[ap.ams_q.py_idx, 0, :, :]

    # Total-evaporation bins: "Aerosol-evaporation gather"
    # (mod_amps_core.F90:616-659) -- number/aerosol-mass/aerosol-soluble
    # to the aerosol group, water-only remainder to vapor. `total_evap`
    # and underflow-eligible (`gather_source`) bins are DISJOINT by
    # construction (underflow only ever fires for `icond_noevp==True`
    # bins, i.e. NOT total_evap), so the two contributions below never
    # overlap for the same (bin, point).
    te_number = np.where(total_evap, con, 0.0)
    te_aero_mass = np.where(total_evap, mass_aero_total, 0.0)
    te_aero_soluble = np.where(total_evap, mass_aero_soluble, 0.0)
    te_water_mass = np.where(total_evap, np.maximum(0.0, mass_total - mass_aero_total), 0.0)

    all_number = te_number + uf_number
    all_aero_mass = te_aero_mass + uf_aero_mass
    all_aero_soluble = te_aero_soluble + uf_aero_soluble
    all_water_mass = te_water_mass + uf_water_mass
    # eps_map from the COMBINED (disjoint-contribution) aerosol mass --
    # falls back to config.eps_ap[0] (diag_pq's own inactive-bin default,
    # core/liquid_diag.py) only when there is genuinely no aerosol mass
    # to derive a ratio from (e.g. an aerosol-free test bin).
    all_eps_map = np.where(
        all_aero_mass > 0.0,
        np.clip(_safe_div(all_aero_soluble, all_aero_mass), 0.0, 1.0),
        float(config.eps_ap[0]),
    )

    if ncat >= 1 and (total_evap.any() or gather_source.any()):
        cat_index_per_bin = _evap_category_index(all_eps_map, ncat)  # (nbins, npoints)
        for i in range(nbins):
            acon = _scatter_add_by_category(acon, all_number[i], cat_index_per_bin[i], ncat)
            amt = _scatter_add_by_category(amt, all_aero_mass[i], cat_index_per_bin[i], ncat)
            ams = _scatter_add_by_category(ams, all_aero_soluble[i], cat_index_per_bin[i], ncat)

    aerosol_values[ap.amt_q.py_idx, 0, :, :] = amt
    aerosol_values[ap.acon_q.py_idx, 0, :, :] = acon
    aerosol_values[ap.ams_q.py_idx, 0, :, :] = ams
    aerosol_out = AerosolState(values=aerosol_values)

    # ---- Vapor exchange (net water mass condensed/evaporated) --------------

    # used_v (G3 section 1 lines 421/436, gathered over all bins) --
    # dMcon for growing/partially-evaporating bins (condensation consumes
    # vapor, dMcon>0 -> vapor decreases; evaporation releases it, dMcon<0
    # -> vapor increases). Total-evaporation/underflow water-only
    # remainders (`all_water_mass`, `-temp_dM` for whole bins == that
    # same water content) are a SEPARATE, additive vapor release --
    # `all_water_mass` already covers both.
    used_v_growth = np.sum(np.where(icond_noevp, dmcon, 0.0), axis=0)
    water_to_vapor = -used_v_growth + np.sum(all_water_mass, axis=0)

    qv_old = get_thermo_prop(thermo_state, ThermoProp.qvv)
    qv_new = qv_old + water_to_vapor
    thermo_values = thermo_state.values.copy()
    qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
    thermo_values[qvv_idx, 0, 0, :] = qv_new
    thermo_out = ThermoState(values=thermo_values)

    return liquid_out, aerosol_out, thermo_out

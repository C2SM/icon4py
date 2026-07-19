# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-List (1982) collisional-breakup fragment-table generator,
`cal_breakfragment`, transcribed VERBATIM from AMPS Fortran (scale_amps
repo) per
docs/superpowers/facts/m2b/breakfragment-full-chain.md ("H3" below, the
Cluster-1 port spec, itself re-confirming/superseding
docs/superpowers/facts/m2/breakup-tables-and-icon4py-m2.md PART A's
"dump, don't port" recommendation -- M2b Task 6's dispatch overrides that
recommendation towards a real port). M2b Task 6.

This is the REAL fill for `core.lookup_tables.BreakupFragmentTables`
(replacing the M1 zero-filled placeholder that lived directly in
`lookup_tables.py`, `allow_placeholder=True`-gated). It lives in its OWN
module, not in `lookup_tables.py` itself, for a genuine layering reason:
the real fill needs `core.collision_kernel.coalescence_efficiency` (H3 §3,
REUSED verbatim -- confirmed the SAME `cal_Coalescence_Efficiency`
token==1 branch H3 §3 quotes) and `core.liquid_diag._terminal_velocity`
(H3 §8.2, REUSED -- H3 confirms this, not the monolithic `diag_pq`, is the
right `%vtm` routine), and BOTH of those modules already import
`lookup_tables` (for `AmpsLuts`/`ColLutAux`) -- a module-level import of
either back into `lookup_tables.py` would be a genuine Python circular
import. `lookup_tables.make_breakup_fragment_tables` is kept as the public
entry point (a thin function-local-import delegator to this module's own
`make_breakup_fragment_tables` -- see that function's own docstring for
why the deferred import is safe), so callers do not need to know this
module exists.

Routines ported (all VERBATIM per H3, `mod_amps_core.F90`/
`mod_amps_utility.F90`/`mod_amps_lib.F90`, scale_amps repo):

* `cal_breakfragment` (driver) -- `mod_amps_lib.F90:1831-2017` (H3 §0, §7).
  Fixed air state (T=278.6795K, PT=850hPa, RH=100%, phase=1), bin-pair
  loop `imin_bk<=i<=imax_bk`, `jmin_bk<=j<=i-1`, calling
  `cal_Coalescence_Efficiency` (REUSED, see above) then
  `cal_breakup_dis_LL` (ported below) per pair.
* `cal_breakup_dis_LL` -- `mod_amps_core.F90:12019-12447` (H3 §2). THE
  core Low & List (1982) filament/sheet/disk fragment-distribution math.
* `cal_sig_sf`/`cal_Hmusig` -- `mod_amps_core.F90:27254-27354` (H3 §4).
  Normal/lognormal-distribution sigma solvers via Brent root-finding.
* `zbrent` (as the generic `_zbrent` helper, objective functions inlined
  as closures rather than the Fortran's `iwhich` integer dispatch --
  H3 §5's own port note: "the port can pass h by value into the objective
  and recompute H at the end") -- `mod_amps_utility.F90:12778-12907`.
  `iwhich==3` (ice-mass path) is dead for `cal_breakfragment` (H3 §5) and
  is NOT ported.
* `getznorm2`/`cumnor` (as `getznorm2`/`_cumnor`) -- standard-normal CDF,
  `mod_amps_utility.F90:369-375` / `:10231-10435` (H3 §6.1, §6.3). `cdfnor`
  itself is not separately ported: `getznorm2` always calls it with
  `which=1`, which H3 §6.2 confirms reduces to a thin `z=(x-mean)/sd;
  call cumnor(z,p,q)` wrapper with `mean=0,sd=1` -- folded directly into
  `getznorm2` here. `dinvnr`/`stvaln`/`eval_pol` (the `which=2/3/4`
  inverse-CDF chain) are NOT on the breakup hot path (H3 §6.4) and are
  NOT ported.

NOT ported (Cluster 2, replaced by a direct computation per H3 §8):
`diag_pq`'s general property-diagnosis stack (`ini_group_MP` ->
`diag_pq` -> `cal_meanmass_vec`/`cal_den_aclen_vec`/
`cal_terminal_vel_vec`/... -- ~2100+ lines, M1 Task 5's original "BLOCKED"
call-tree). Only the two closed-form scalars the breakup math actually
reads (`%len`, `%vtm`) are computed directly:

* `%len` (H3 §8.1): `len_i = (mean_mass_i / coedpi6)**(1/3)` [cm], the
  SAME geometric formula `cal_breakup_dis_LL` already uses internally on
  bin BOUNDARIES (`D=(binb/coedpi6)**(1/3)`) -- self-consistent by
  construction. `mean_mass_i` (the bin's representative per-drop mass) is
  NOT simply a function of the bin boundaries alone, despite H3 §8.1's own
  phrasing ("deterministic from binb/srat_r/minmass_r") -- reading
  `ini_group_MP` (`class_Group.F90:10991-11063`, the routine
  `cal_breakfragment` actually calls, `mod_amps_lib.F90:1941`) directly
  (this task's dispatch authorizes reading named-but-not-quoted Fortran
  when H3's own text under-specifies a formula) shows `%con`/`%mass(1)`
  per bin are seeded from a Marshall-Palmer-type analytic rain spectrum
  (`rr=54mm/hr` rain rate, `lambda=4.1*rr**-0.21`, `n_0=8.0e3`), integrated
  over each bin's mass-boundary-derived melt-diameter interval -- NOT a
  bin-boundary geometric mean. `mean_mass_i = mass(1)_i/con_i`
  (`cal_meanmass_vec`, `class_Mass_Bin.F90:1840-1849`, with `diag_pq`'s
  own `con>1e-30 and mass(1)>1e-30` guard, `mean_mass=0` otherwise --
  `_liquid_mean_mass`/`_ini_group_mp_liquid` below port BOTH exactly).
  This is a genuine, documented gap-fill beyond H3's own text -- see the
  M2b Task 6 report for the full call-out.
* `%vtm` (H3 §8.2): `core.liquid_diag._terminal_velocity`, called at the
  SAME fixed air state (`_fixed_air_state` below ports the scalar subset
  of `make_Thermo_Var3_2`, `class_Thermo_Var.F90:309-369`, that
  `_terminal_velocity` needs: `P` [CGS, `pres*10`], `den_a`, `d_vis`,
  `sig_wa` -- H3 §8.2's own "one fixed ThermoState" framing).

Units: CGS throughout (grams, cm, cm/s, erg), except where
`coalescence_efficiency` (REUSED from `core.collision_kernel`, itself a
verbatim port of `cal_Coalescence_Efficiency`) returns SI-valued
`(CKE, D_L, D_S, S_T, S_C)` -- see that module's own docstring; this
module passes those SI values straight into `cal_breakup_dis_ll`, which
(like the Fortran) re-converts `D_L`/`D_S` to CGS internally (`*100`).
float64 throughout.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import (
    bin_grid,
    collision_kernel,
    lookup_tables,
    thermo as thermo_fn,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import (
    LiquidDiag,
    _terminal_velocity,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import ThermoProp, ThermoState


# ---------------------------------------------------------------------------
# Section 6: getznorm2 / cumnor -- standard-normal CDF, Cody rational
# approximation, VERBATIM (H3 §6.1, §6.3), vectorized (numpy array-in,
# array-out; also used scalar-in-scalar-out from the zbrent objectives).
# ---------------------------------------------------------------------------

_CUMNOR_A = np.array(
    [
        2.2352520354606839287e0,
        1.6102823106855587881e2,
        1.0676894854603709582e3,
        1.8154981253343561249e4,
        6.5682337918207449113e-2,
    ]
)
_CUMNOR_B = np.array(
    [
        4.7202581904688241870e1,
        9.7609855173777669322e2,
        1.0260932208618978205e4,
        4.5507789335026729956e4,
    ]
)
_CUMNOR_C = np.array(
    [
        3.9894151208813466764e-1,
        8.8831497943883759412e0,
        9.3506656132177855979e1,
        5.9727027639480026226e2,
        2.4945375852903726711e3,
        6.8481904505362823326e3,
        1.1602651437647350124e4,
        9.8427148383839780218e3,
        1.0765576773720192317e-8,
    ]
)
_CUMNOR_D = np.array(
    [
        2.2266688044328115691e1,
        2.3538790178262499861e2,
        1.5193775994075548050e3,
        6.4855582982667607550e3,
        1.8615571640885098091e4,
        3.4900952721145977266e4,
        3.8912003286093271411e4,
        1.9685429676859990727e4,
    ]
)
_CUMNOR_P = np.array(
    [
        2.1589853405795699e-1,
        1.274011611602473639e-1,
        2.2235277870649807e-2,
        1.421619193227893466e-3,
        2.9112874951168792e-5,
        2.307344176494017303e-2,
    ]
)
_CUMNOR_Q = np.array(
    [
        1.28426009614491121e0,
        4.68238212480865118e-1,
        6.59881378689285515e-2,
        3.78239633202758244e-3,
        7.29751555083966205e-5,
    ]
)
_ROOT32 = 5.656854248e0
_SIXTEN = 16.0
_SQRPI = 3.9894228040143267794e-1
_THRSH = 0.66291e0
_EPS_CUMNOR = float(np.finfo(np.float64).eps) * 0.5
_TINY_CUMNOR = float(np.finfo(np.float64).tiny)


def _cumnor(arg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`cumnor(arg) -> (cum, ccum)`, VERBATIM Cody rational-approximation
    normal CDF, `mod_amps_utility.F90:10231-10435` (H3 §6.3). Vectorized:
    `arg` may be any shape; the three branches (`|x|<=thrsh`,
    `<=root32`, `>root32`) are evaluated for every element (wrapped in
    `np.errstate` to silence the expected overflow/divide warnings from
    branches later discarded by `np.select`) and combined via masks --
    numerically equivalent to the Fortran's per-call branch selection.
    """
    x = np.asarray(arg, dtype=np.float64)
    y = np.abs(x)

    low = y <= _THRSH
    mid = (~low) & (y <= _ROOT32)
    high = ~low & ~mid

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # -- low branch: |x| <= thrsh --
        xsq_low = np.where(y > _EPS_CUMNOR, x * x, 0.0)
        xnum = _CUMNOR_A[4] * xsq_low
        xden = xsq_low.copy()
        for i in range(3):
            xnum = (xnum + _CUMNOR_A[i]) * xsq_low
            xden = (xden + _CUMNOR_B[i]) * xsq_low
        cum_low_raw = x * (xnum + _CUMNOR_A[3]) / (xden + _CUMNOR_B[3])
        cum_low = 0.5 + cum_low_raw
        ccum_low = 0.5 - cum_low_raw

        # -- mid branch: thrsh < |x| <= root32 --
        y_mid = np.where(mid, y, 1.0)
        xnum_m = _CUMNOR_C[8] * y_mid
        xden_m = y_mid.copy()
        for i in range(7):
            xnum_m = (xnum_m + _CUMNOR_C[i]) * y_mid
            xden_m = (xden_m + _CUMNOR_D[i]) * y_mid
        base_cum_mid = (xnum_m + _CUMNOR_C[7]) / (xden_m + _CUMNOR_D[7])
        xsq_mid = np.trunc(y_mid * _SIXTEN) / _SIXTEN
        del_mid = (y_mid - xsq_mid) * (y_mid + xsq_mid)
        base_cum_mid = np.exp(-xsq_mid * xsq_mid * 0.5) * np.exp(-del_mid * 0.5) * base_cum_mid
        base_ccum_mid = 1.0 - base_cum_mid
        pos_mid = x > 0.0
        cum_mid = np.where(pos_mid, base_ccum_mid, base_cum_mid)
        ccum_mid = np.where(pos_mid, base_cum_mid, base_ccum_mid)

        # -- high branch: |x| > root32 --
        y_high = np.where(high, y, 1.0)
        x_high = np.where(high, x, 1.0)
        xsq_h = 1.0 / (x_high * x_high)
        xnum_h = _CUMNOR_P[5] * xsq_h
        xden_h = xsq_h.copy()
        for i in range(4):
            xnum_h = (xnum_h + _CUMNOR_P[i]) * xsq_h
            xden_h = (xden_h + _CUMNOR_Q[i]) * xsq_h
        base_cum_high = xsq_h * (xnum_h + _CUMNOR_P[4]) / (xden_h + _CUMNOR_Q[4])
        base_cum_high = (_SQRPI - base_cum_high) / y_high
        xsq_h2 = np.trunc(x_high * _SIXTEN) / _SIXTEN
        del_h = (x_high - xsq_h2) * (x_high + xsq_h2)
        base_cum_high = np.exp(-xsq_h2 * xsq_h2 * 0.5) * np.exp(-del_h * 0.5) * base_cum_high
        base_ccum_high = 1.0 - base_cum_high
        pos_high = x_high > 0.0
        cum_high = np.where(pos_high, base_ccum_high, base_cum_high)
        ccum_high = np.where(pos_high, base_cum_high, base_ccum_high)

    cum = np.select([low, mid, high], [cum_low, cum_mid, cum_high])
    ccum = np.select([low, mid, high], [ccum_low, ccum_mid, ccum_high])

    cum = np.where(cum < _TINY_CUMNOR, 0.0, cum)
    ccum = np.where(ccum < _TINY_CUMNOR, 0.0, ccum)
    return cum, ccum


def getznorm2(x: float | np.ndarray) -> float | np.ndarray:
    """Standard-normal CDF `Phi(x)`, `getznorm2` (`mod_amps_utility.F90:
    369-375`, H3 §6.1): `mean=0, sd=1; call cdfnor(1, p, q, x, mean, sd,
    ...)`, which (H3 §6.2, `which==1`) reduces to `cumnor(x, p, q)`
    directly -- `p` (== `cum` here) is `getznorm2`'s return value.

    Returns a plain `float` for scalar input (so the `zbrent` objective
    closures below get ordinary Python floats to do arithmetic/compare
    against `-999.9` with), or an `np.ndarray` matching `x`'s shape for
    array input (the vectorized per-bin distribution formulas in
    `cal_breakup_dis_ll`).
    """
    arr = np.asarray(x, dtype=np.float64)
    cum, _ccum = _cumnor(arr)
    if arr.ndim == 0:
        return float(cum)
    return cum


# ---------------------------------------------------------------------------
# Section 5: zbrent -- Brent root-finder, VERBATIM (H3 §5,
# mod_amps_utility.F90:12778-12907), generalized to take a plain Python
# objective callable instead of the Fortran's `iwhich` integer dispatch
# (H3 §5's own port note). Returns the `-999.9` sentinel on a
# non-bracketing `[x1, x2]`, exactly as the Fortran does (no exception).
# ---------------------------------------------------------------------------

_ZBRENT_SENTINEL = -999.9
_ZBRENT_ITMAX = 100
_ZBRENT_EPS = 3.0e-8


def _zbrent(func: Callable[[float], float], x1: float, x2: float, tol: float) -> float:
    """Brent's method, VERBATIM (H3 §5). `func` plays the role of the
    Fortran's internal `contains`-function `func` (there dispatched on
    `iwhich`; here just a closure the caller builds per-objective, see
    `cal_sig_sf`/`cal_hmusig` below)."""
    a, b = x1, x2
    fa, fb = func(a), func(b)
    if (fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0):
        return _ZBRENT_SENTINEL

    c, fc = b, fb
    d = e = b - a
    for _iter in range(_ZBRENT_ITMAX):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            # Fortran's sequential a=b;b=c;c=a;fa=fb;fb=fc;fc=fa reduces to
            # this simultaneous rotation (c ends up equal to the OLD b, not
            # the old a) -- see this module's own derivation in the M2b
            # Task 6 report.
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol1 = 2.0 * _ZBRENT_EPS * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a, fa = b, fb
        b = b + d if abs(d) > tol1 else b + math.copysign(tol1, xm)
        fb = func(b)

    return b  # max iterations exceeded -- Fortran returns the last `b` too


# ---------------------------------------------------------------------------
# Section 4: cal_sig_sf / cal_Hmusig -- sigma solvers, VERBATIM (H3 §4,
# mod_amps_core.F90:27254-27354).
# ---------------------------------------------------------------------------

_SIG_ITER = 10  # ITER, both routines
_ZBRENT_TOL = 1.0e-4


def cal_sig_sf(d0: float, r: float, nx: float, h_s: float, mu_s: float) -> float:
    """Normal-distribution sigma via Brent (H3 §4, `cal_sig_sf`). `zbrent`
    is called with `iwhich==1`'s objective inlined as a closure."""
    coedsq2p = float(AmpsConst.coedsq2p)
    sig_s = 1.0 / (h_s * coedsq2p)
    if r < 1.0e-20:
        return sig_s

    for i in range(1, _SIG_ITER + 1):

        def objective(
            x: float, nx: float = nx, h_s: float = h_s, d0: float = d0, mu_s: float = mu_s
        ) -> float:
            phi = min(0.99999, getznorm2((d0 - mu_s) / max(x, 1.0e-20)))
            return x - nx / h_s / coedsq2p / (1.0 - phi)

        trial = _zbrent(objective, sig_s * 1.0e-8, sig_s * 10.0, _ZBRENT_TOL)
        if trial == _ZBRENT_SENTINEL:
            sig_s = 1.0 / (h_s * coedsq2p) / 10.0**i
        else:
            sig_s = trial
            break
    return sig_s


def cal_hmusig(
    d0: float, r: float, lin_mu: float, p_mode: float, nx: float
) -> tuple[float, float, float]:
    """Lognormal-distribution H/mu/sigma via Brent (H3 §4, `cal_Hmusig`).
    `zbrent` is called with `iwhich==2`'s objective inlined as a closure
    (its `h=p_mode*mu*exp(0.5*x**2)` in-loop mutation is transient -- H3
    §4's own port note: computed locally inside the objective, `H`
    recomputed once more from the converged `sig` after the loop, exactly
    matching the Fortran's own post-loop `H=P_mode*lin_mu*exp(0.5*sig**2)`
    line)."""
    coedsq2p = float(AmpsConst.coedsq2p)
    sig = 10.0 * lin_mu
    if nx <= 1.0e-20 or p_mode <= 1.0e-20 or r <= 1.0e-20:
        return 0.0, 0.0, sig

    for _i in range(1, _SIG_ITER + 1):
        osig = sig

        def objective(
            x: float, p_mode: float = p_mode, nx: float = nx, d0: float = d0, lin_mu: float = lin_mu
        ) -> float:
            h_local = p_mode * lin_mu * math.exp(0.5 * x * x)
            c1 = math.log(lin_mu) + x * x
            phi = min(0.99999999, getznorm2((math.log(d0) - c1) / max(x, 1.0e-20)))
            return x - nx / h_local / coedsq2p / (1.0 - phi)

        trial = _zbrent(objective, sig * 1.0e-5, sig * 10.0, _ZBRENT_TOL)
        if trial == _ZBRENT_SENTINEL:
            sig = osig * 1.5
        else:
            sig = trial
            break

    h = p_mode * lin_mu * math.exp(0.5 * sig * sig)
    mu = math.log(lin_mu) + sig * sig
    return h, mu, sig


# ---------------------------------------------------------------------------
# Section 2: cal_breakup_dis_LL -- THE core Low & List (1982) fragment
# distribution, VERBATIM (H3 §2, mod_amps_core.F90:12019-12447).
# ---------------------------------------------------------------------------

_CKE0 = 8.93e-7
_W0 = 0.86
_APP = 1.02e4
_BPP = 2.83
_D_0_CM = 0.01  # low-diameter breakup cutoff, cm


def _znorm_arr(a: np.ndarray) -> np.ndarray:
    """`getznorm2` restricted to the array-in/array-out case (asserted),
    for the vectorized per-bin formulas below."""
    out = getznorm2(a)
    assert isinstance(out, np.ndarray)
    return out


def _lognormal_fragment_number_mass(  # noqa: PLR0917 [too-many-positional-arguments]
    d_log1: np.ndarray,
    d_log2: np.ndarray,
    log_d_coal: float,
    log_d0: float,
    h_ln: float,
    mu_ln: float,
    sig_ln: float,
    coedpi6: float,
    coedsq2p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The lognormal FRAGMENT-drop number/mass distribution shared by all
    three breakup modes (H3 §2: `n_f(1,ibin)`/`m_f(1,ibin)` and their
    `n_s`/`m_s`, `n_d`/`m_d` siblings -- identical formula shape, only the
    `H_ln`/`mu_ln`/`sig_ln` triple differs per mode)."""
    lo = np.maximum(d_log1, log_d0)
    hi = np.minimum(d_log2, log_d_coal)
    n = np.maximum(
        0.0,
        h_ln
        * sig_ln
        * coedsq2p
        * (_znorm_arr((hi - mu_ln) / sig_ln) - _znorm_arr((lo - mu_ln) / sig_ln)),
    )
    m = np.maximum(
        0.0,
        coedpi6
        * h_ln
        * sig_ln
        * coedsq2p
        * math.exp(4.5 * sig_ln * sig_ln + 3.0 * mu_ln)
        * (
            _znorm_arr((hi - mu_ln) / sig_ln - 3.0 * sig_ln)
            - _znorm_arr((lo - mu_ln) / sig_ln - 3.0 * sig_ln)
        ),
    )
    return n, m


def _normal_parent_number_mass(  # noqa: PLR0917 [too-many-positional-arguments]
    d_1: np.ndarray,
    d_2: np.ndarray,
    h: float,
    sig: float,
    mu: float,
    coedpi6: float,
    coedsq2p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The normal PARENT-drop (small or large) number/mass distribution
    shared by filament/sheet/disk breakup (H3 §2: `n_f(2/3,ibin)`/
    `m_f(2/3,ibin)` and their `n_s`/`m_s`, `n_d`/`m_d` siblings -- same
    formula shape, only `H`/`sig`/`mu` differ per mode/parent)."""
    n = h * sig * coedsq2p * (_znorm_arr((d_2 - mu) / sig) - _znorm_arr((d_1 - mu) / sig))
    x1 = (d_1 - mu) / sig
    x2 = (d_2 - mu) / sig
    c1 = sig * (sig * sig * (x1 * x1 + 2.0) + 3.0 * mu * (sig * x1 + mu)) / coedsq2p
    c2 = sig * (sig * sig * (x2 * x2 + 2.0) + 3.0 * mu * (sig * x2 + mu)) / coedsq2p
    a1 = mu * (3.0 * sig * sig + mu * mu)
    m = (
        coedpi6
        * h
        * sig
        * coedsq2p
        * (
            -c2 * np.exp(-x2 * x2 / 2.0)
            + c1 * np.exp(-x1 * x1 / 2.0)
            + a1 * (_znorm_arr(x2) - _znorm_arr(x1))
        )
    )
    return n, m


@dataclasses.dataclass(frozen=True)
class _BreakupShapeParams:
    """Steps 1-3 of `cal_breakup_dis_LL` (H3 §2): the mode fractions
    (`R_f`/`R_s`/`R_d`), average fragment counts (`F_f`/`F_s`/`F_d`), and
    parent-distribution `(H,mu,sig)` triples for filament-large,
    filament-small, sheet-large, disk-large -- everything step 4 (the
    fragment lognormal parameters) and the per-bin loop need, bundled to
    keep `cal_breakup_dis_ll` itself under the statement-count budget."""

    r_f: float
    r_s: float
    r_d: float
    f_f: float
    f_s: float
    f_d: float
    h_lf: float
    mu_lf: float
    sig_lf: float
    h_sf: float
    mu_sf: float
    sig_sf: float
    h_ls: float
    mu_ls: float
    sig_ls: float
    h_ld: float
    mu_ld: float
    sig_ld: float


def _compute_breakup_shape_params(
    d_l: float, d_s: float, s_t: float, s_c: float, cke: float
) -> _BreakupShapeParams:
    """H3 §2 steps "1. Determine the fraction of collision-breakup types",
    "2. calculate average number of fragments per a collision", and
    "3. Calculate the parameters for parents distribution" -- VERBATIM."""
    sq_twod = math.sqrt(2.0)
    w2 = cke / s_t

    # 1. fraction of collision-breakup types.
    r_f = 1.11e-4 * cke ** (-0.654) if cke >= _CKE0 else 1.0
    r_s = 0.685 * (1.0 - math.exp(-1.63 * (w2 - _W0))) if w2 >= _W0 else 0.0
    if r_s + r_f > 1.0:
        r_s = 1.0 - r_f
        r_d = 0.0
    else:
        r_d = max(1.0 - r_f - r_s, 0.0)

    # 2. average number of fragments per collision.
    f_f = (
        (-2.25e4 * (d_l - 0.403) * (d_l - 0.403) - 37.9) * d_s**2.5
        + 9.67 * (d_l + 0.170) * (d_l + 0.170)
        + 4.95
    )
    f_f = max(2.0, min(f_f, _APP * d_s**_BPP + 2.0))
    f_s = max(5.0 * (2.0 * getznorm2(sq_twod * (s_t - 2.53e-6) / 1.85e-6) - 1.0) + 6.0, 2.0)
    f_d = max(297.5 + 23.76 * math.log(cke), 2.0)

    # 3. parent-distribution parameters.
    h_lf = 50.8 * d_l ** (-0.718)
    h_sf = 4.18 * d_s ** (-1.17)
    h_ls = 100.0 * math.exp(-3.25 * d_s)
    h_ld = 1.58e-5 * cke ** (-1.22)
    mu_lf = d_l
    mu_sf = d_s
    mu_ls = d_l
    mu_ld = d_l * (1.0 - math.exp(-3.70 * (3.10 - cke / s_c)))
    sig_lf = cal_sig_sf(_D_0_CM, r_f, 1.0, h_lf, mu_lf)
    sig_sf = cal_sig_sf(_D_0_CM, r_f, 1.0, h_sf, mu_sf)
    sig_ls = cal_sig_sf(_D_0_CM, r_s, 1.0, h_ls, mu_ls)
    sig_ld = cal_sig_sf(_D_0_CM, r_d, 1.0, h_ld, mu_ld)

    return _BreakupShapeParams(
        r_f=r_f,
        r_s=r_s,
        r_d=r_d,
        f_f=f_f,
        f_s=f_s,
        f_d=f_d,
        h_lf=h_lf,
        mu_lf=mu_lf,
        sig_lf=sig_lf,
        h_sf=h_sf,
        mu_sf=mu_sf,
        sig_sf=sig_sf,
        h_ls=h_ls,
        mu_ls=mu_ls,
        sig_ls=sig_ls,
        h_ld=h_ld,
        mu_ld=mu_ld,
        sig_ld=sig_ld,
    )


@dataclasses.dataclass(frozen=True)
class _FragmentLognormalParams:
    """Step 4 of `cal_breakup_dis_LL` (H3 §2): the lognormal
    fragment-drop distribution `(H,mu,sig)` triples, one per breakup
    mode."""

    h_lnf: float
    mu_lnf: float
    sig_lnf: float
    h_lns: float
    mu_lns: float
    sig_lns: float
    h_lnd: float
    mu_lnd: float
    sig_lnd: float


def _compute_fragment_lognormal_params(
    d_l: float, d_s: float, shape: _BreakupShapeParams
) -> _FragmentLognormalParams:
    """H3 §2 step "4. calculate the parameters for fragment
    distribution" -- VERBATIM (the `P_mode` piecewise formula differs per
    mode; `cal_hmusig` -> `(H,mu,sig)` per mode)."""
    lin_mu_lnf = 0.241 * d_s + 0.0129
    if d_s <= _D_0_CM:
        p_mode_f = 1.68e5 * d_s**2.33
    elif d_s >= 1.2 * _D_0_CM:
        p_mode_f = (
            (43.4 * (d_l + 1.81) * (d_l + 1.81) - 159.0) / d_s
            - 3870.0 * (d_l - 0.285) * (d_l - 0.285)
            - 58.1
        )
    else:
        dum = (d_s - _D_0_CM) / (0.2 * _D_0_CM)
        p_mode_f = dum * (1.68e5 * d_s**2.33) + (1.0 - dum) * (
            (43.4 * (d_l + 1.81) * (d_l + 1.81) - 159.0) / d_s
            - 3870.0 * (d_l - 0.285) * (d_l - 0.285)
            - 58.1
        )
    h_lnf, mu_lnf, sig_lnf = cal_hmusig(_D_0_CM, shape.r_f, lin_mu_lnf, p_mode_f, shape.f_f - 2.0)

    lin_mu_lns = 0.254 * d_s**0.413 * math.exp((3.53 * d_s - 2.51) * (d_l - d_s))
    p_mode_s = 0.23 * d_s ** (-3.93) * d_l ** (14.2 * math.exp(-17.2 * d_s))
    h_lns, mu_lns, sig_lns = cal_hmusig(_D_0_CM, shape.r_s, lin_mu_lns, p_mode_s, shape.f_s - 1.0)

    lin_mu_lnd = math.exp((-17.4 * d_s - 0.671) * (d_l - d_s)) * d_s
    if (d_l - d_s) < 0.5 and 0.007 * d_s ** (-2.54) > 100.0:
        p_mode_d = 0.0
    else:
        p_mode_d = 8.84 * d_s ** (-2.52) * (d_l - d_s) ** (0.007 * d_s ** (-2.54))
    h_lnd, mu_lnd, sig_lnd = cal_hmusig(_D_0_CM, shape.r_d, lin_mu_lnd, p_mode_d, shape.f_d - 1.0)

    return _FragmentLognormalParams(
        h_lnf=h_lnf,
        mu_lnf=mu_lnf,
        sig_lnf=sig_lnf,
        h_lns=h_lns,
        mu_lns=mu_lns,
        sig_lns=sig_lns,
        h_lnd=h_lnd,
        mu_lnd=mu_lnd,
        sig_lnd=sig_lnd,
    )


def cal_breakup_dis_ll(  # noqa: PLR0917 [too-many-positional-arguments]
    binb: np.ndarray,
    nbin: int,
    d_l_m: float,
    d_s_m: float,
    s_t_j: float,
    s_c_j: float,
    cke_j: float,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """`cal_breakup_dis_LL`, VERBATIM (H3 §2). `binb` is the liquid bin
    MASS-boundary array (CGS grams, length `nbin+1`); `d_l_m`/`d_s_m` (SI
    meters), `s_t_j`/`s_c_j`/`cke_j` (SI Joules) are `coalescence_efficiency`'s
    own `(D_L, D_S, S_T, S_C, CKE)` outputs for this bin pair, UNCONVERTED
    (this function re-converts `D_L`/`D_S` to CGS internally via `*100`,
    exactly as the Fortran does).

    Returns `None` if `D_coal <= D_0` (the Fortran's own early `return`,
    `H3 §2` line `if(D_coal<=D_0) return` -- BEFORE `bu_tmass`/`bu_fd` are
    written, so the caller must skip writing entirely, not write a zero;
    in practice unreachable from `make_breakup_fragment_tables`'s own
    bin-pair loop, since both `i` and `j` are already `>= jmin_bk`, i.e.
    both individual bin diameters already clear `D_0`, so their coalesced
    diameter trivially does too -- kept for fidelity/robustness anyway).
    Otherwise returns `(m_coal, frag_mass, frag_con)`: `m_coal` (g, the
    coalesced-drop mass -> `bu_tmass`), `frag_mass`/`frag_con` (each
    shape `(nbin,)`, g / cm^-3 -> `bu_fd[0, :]` / `bu_fd[1, :]` for this
    pair's `kk` range).
    """
    coedpi6 = float(AmpsConst.coedpi6)
    coedsq2p = float(AmpsConst.coedsq2p)

    d_l = d_l_m * 100.0
    d_s = d_s_m * 100.0
    s_c = s_c_j
    s_t = s_t_j
    cke = cke_j

    d_coal = (d_l**3.0 + d_s**3.0) ** (1.0 / 3.0)
    m_coal = coedpi6 * d_coal**3.0
    if d_coal <= _D_0_CM:
        return None

    shape = _compute_breakup_shape_params(d_l, d_s, s_t, s_c, cke)
    r_f, r_s, r_d = shape.r_f, shape.r_s, shape.r_d
    h_lf, mu_lf, sig_lf = shape.h_lf, shape.mu_lf, shape.sig_lf
    h_sf, mu_sf, sig_sf = shape.h_sf, shape.mu_sf, shape.sig_sf
    h_ls, mu_ls, sig_ls = shape.h_ls, shape.mu_ls, shape.sig_ls
    h_ld, mu_ld, sig_ld = shape.h_ld, shape.mu_ld, shape.sig_ld

    frag = _compute_fragment_lognormal_params(d_l, d_s, shape)
    h_lnf, mu_lnf, sig_lnf = frag.h_lnf, frag.mu_lnf, frag.sig_lnf
    h_lns, mu_lns, sig_lns = frag.h_lns, frag.mu_lns, frag.sig_lns
    h_lnd, mu_lnd, sig_lnd = frag.h_lnd, frag.mu_lnd, frag.sig_lnd

    # Per-bin (vectorized over ibin=1..nbin) number/mass distributions.
    # NOTE: the Fortran's own `Do ibin=max(i,j)+1,g_1%N_BIN+1 ... ibin_coal
    # = ...; exit` block (H3 §2, right before the main per-bin loop) only
    # ever ASSIGNS `ibin_coal`, never reads it again anywhere in the
    # quoted routine (the commented-out `!!c Do ibin=1,ibin_coal` was
    # replaced by the ACTIVE `Do ibin=1,g_1%N_BIN`, which loops every bin
    # unconditionally) -- genuinely dead code, not ported.
    binb_lo = binb[:nbin]
    binb_hi = binb[1 : nbin + 1]
    d_log2 = np.log((binb_hi / coedpi6) ** (1.0 / 3.0))
    d_log1 = np.log((binb_lo / coedpi6) ** (1.0 / 3.0))
    d_2 = (binb_hi / coedpi6) ** (1.0 / 3.0)
    d_1 = (binb_lo / coedpi6) ** (1.0 / 3.0)

    log_d_coal = math.log(d_coal)
    log_d0 = math.log(_D_0_CM)

    # ---- filament breakup ----
    n_f1, m_f1 = _lognormal_fragment_number_mass(
        d_log1, d_log2, log_d_coal, log_d0, h_lnf, mu_lnf, sig_lnf, coedpi6, coedsq2p
    )
    n_f2, m_f2 = _normal_parent_number_mass(d_1, d_2, h_sf, sig_sf, mu_sf, coedpi6, coedsq2p)
    n_f3, m_f3 = _normal_parent_number_mass(d_1, d_2, h_lf, sig_lf, mu_lf, coedpi6, coedsq2p)

    # ---- sheet breakup (no small-parent term: n_s(2,ibin)=m_s(2,ibin)=0) ----
    n_s1, m_s1 = _lognormal_fragment_number_mass(
        d_log1, d_log2, log_d_coal, log_d0, h_lns, mu_lns, sig_lns, coedpi6, coedsq2p
    )
    n_s2 = m_s2 = np.zeros(nbin)
    n_s3, m_s3 = _normal_parent_number_mass(d_1, d_2, h_ls, sig_ls, mu_ls, coedpi6, coedsq2p)

    # ---- disk breakup (no small-parent term: n_d(2,ibin)=m_d(2,ibin)=0) ----
    n_d1, m_d1 = _lognormal_fragment_number_mass(
        d_log1, d_log2, log_d_coal, log_d0, h_lnd, mu_lnd, sig_lnd, coedpi6, coedsq2p
    )
    n_d2 = m_d2 = np.zeros(nbin)
    n_d3, m_d3 = _normal_parent_number_mass(d_1, d_2, h_ld, sig_ld, mu_ld, coedpi6, coedsq2p)

    dmass = r_f * (m_f1 + m_f2 + m_f3) + r_s * (m_s1 + m_s2 + m_s3) + r_d * (m_d1 + m_d2 + m_d3)
    dcon = r_f * (n_f1 + n_f2 + n_f3) + r_s * (n_s1 + n_s2 + n_s3) + r_d * (n_d1 + n_d2 + n_d3)

    invalid = (dcon < 1.0e-100) | (dmass < 1.0e-100)
    dcon = np.where(invalid, 0.0, dcon)
    dmass = np.where(invalid, 0.0, dmass)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dcon > 0.0, dmass / np.where(dcon > 0.0, dcon, 1.0), 0.0)
    out_of_range = (~invalid) & ((ratio > binb_hi) | (ratio < binb_lo))
    dmass = np.where(out_of_range, (binb_hi + binb_lo) * 0.5 * dcon, dmass)

    mrat = m_coal / np.sum(dmass)
    frag_mass = mrat * dmass
    frag_con = mrat * dcon

    return m_coal, frag_mass, frag_con


# ---------------------------------------------------------------------------
# Section 8: %len / %vtm direct computation (H3 §8), replacing diag_pq.
# ---------------------------------------------------------------------------

_RR_FIXED_MM_HR = 54.0  # rain rate, ini_group_mp literal (class_Group.F90:11029)
_N0_FIXED = 8.0e3  # m^-3 mm^-1, ini_group_mp literal


def _liquid_mean_mass(binb: np.ndarray, nbin: int) -> np.ndarray:
    """Per-bin representative mass `mean_mass_i` [g], VERBATIM
    `ini_group_mp` (`class_Group.F90:10991-11063`, read directly per this
    module's own docstring gap-fill note) + `cal_meanmass_vec`
    (`class_Mass_Bin.F90:1840-1849`) + `diag_pq`'s own
    `con>1e-30 and mass(1)>1e-30` activity guard (`core/liquid_diag.py`'s
    own `_mean_mass_and_active` docstring quotes the identical guard for
    the SAME reason -- not re-imported here, to keep this module's only
    `liquid_diag` dependency the single sanctioned `_terminal_velocity`
    call, per the module docstring's layering note).

    `binb` is the liquid bin MASS-boundary array (CGS grams, length
    `nbin+1`); returns `(nbin,)`, 0.0 for bins where `ini_group_mp`'s
    analytic spectrum evaluates to (numerically) zero mass/concentration
    (the Fortran's own `icond1=1` inactive-bin case -- physically the
    haze-scale bins far below the rain spectrum's mode).
    """
    lam = 4.1 * _RR_FIXED_MM_HR ** (-0.21)
    pi = float(AmpsConst.PI)

    binb_lo = binb[:nbin]
    binb_hi = binb[1 : nbin + 1]
    # melt-equivalent diameter at bin boundaries, mm (den=1 g/cm^3 CGS
    # folded into the `6*mass/pi` form, then *10 cm->mm).
    d1 = (6.0 * binb_lo / pi) ** (1.0 / 3.0) * 10.0
    d2 = (6.0 * binb_hi / pi) ** (1.0 / 3.0) * 10.0
    t1 = lam * d1
    t2 = lam * d2

    with np.errstate(over="ignore", under="ignore"):
        con = (_N0_FIXED / lam) * (np.exp(-lam * d1) - np.exp(-lam * d2))
        mass = (
            (
                (t1**3.0 + 3.0 * t1**2.0 + 6.0 * t1 + 6.0) * np.exp(-t1)
                - (t2**3.0 + 3.0 * t2**2.0 + 6.0 * t2 + 6.0) * np.exp(-t2)
            )
            * (pi / 6.0)
            * 0.001
            * (_N0_FIXED / lam**4.0)
        )
    con = con / 1.0e6  # m^-3 -> cm^-3
    mass = mass / 1.0e6  # g/m^3 -> g/cm^3 (per-bin total mass concentration)

    valid = (con > 1.0e-30) & (mass > 1.0e-30)
    mean_mass = np.where(valid, mass / np.where(valid, con, 1.0), 0.0)
    return mean_mass


@dataclasses.dataclass(frozen=True)
class _FixedAirState:
    """The single scalar air/thermo state `cal_breakfragment` uses (H3
    §0, §7): `T=278.6795 K, PT=850 hPa, RH=100%, phase=1 (water)`. Fields
    are exactly the subset `liquid_diag._terminal_velocity` needs, CGS
    (H3 §8.2's "one fixed ThermoState" framing)."""

    t_k: float
    p_cgs: float  # dyn/cm^2
    den_a: float  # g/cm^3, dry-air density
    d_vis: float  # g/cm/s, dynamic viscosity
    sig_wa: float  # erg/cm^2 == dyn/cm, surface tension of water


_T_FIXED_K = 278.6795
_PT_FIXED_PA = 850.0e2  # 850 hPa, SI Pa (mod_amps_lib.F90:1917-1918's own T/PT literals)
_RH_FIXED_PCT = 100.0


def _fixed_air_state() -> _FixedAirState:
    """The scalar subset of `make_Thermo_Var3_2` (`class_Thermo_Var.F90:
    309-369`) that `_terminal_velocity` needs, evaluated at
    `cal_breakfragment`'s fixed air state:

        th_var%P = pres*10                                   ! SI Pa -> CGS dyn/cm^2
        th_var%e_sat(1) = get_sat_vapor_pres_lk(1, T, ...)    ! thermo_fn.esat_lk
        th_var%rv_sat(1) = Rdvchiarui*e_sat(1)/max(P-e_sat(1), e_sat(1))
        th_var%rv = RH*rv_sat(1)/100                          ! RH=100 -> rv=rv_sat(1)
        th_var%e = rv*P/(Rdvchiarui+rv)
        th_var%den_a = M_a*(P-e)/(R_u*T)
    """
    t = _T_FIXED_K
    p_cgs = _PT_FIXED_PA * 10.0
    d_vis = float(thermo_fn.dynamic_viscosity(t))
    sig_wa = float(thermo_fn.sfc_tension(t))

    estbar, esitbar = thermo_fn.make_esat_tables()
    e_sat1 = float(thermo_fn.esat_lk(1, t, estbar, esitbar))

    rdvchiarui = float(AmpsConst.Rdvchiarui)
    rv_sat1 = rdvchiarui * e_sat1 / max(p_cgs - e_sat1, e_sat1)
    rv = _RH_FIXED_PCT * rv_sat1 / 100.0
    e = rv * p_cgs / (rdvchiarui + rv)

    m_a = float(AmpsConst.M_a)
    r_u = float(AmpsConst.R_u)
    den_a = m_a * (p_cgs - e) / (r_u * t)

    return _FixedAirState(t_k=t, p_cgs=p_cgs, den_a=den_a, d_vis=d_vis, sig_wa=sig_wa)


def _make_liquid_diag_for_coalescence(length: np.ndarray, vtm: np.ndarray) -> LiquidDiag:
    """A `LiquidDiag` with only `length`/`terminal_velocity` populated
    meaningfully -- `collision_kernel.coalescence_efficiency` reads
    nothing else (see that function's own docstring). Every other field
    is a harmless placeholder, mirroring the same convention
    `test_collision_kernel.py::_make_diag` already uses."""
    shape = (length.shape[0], 1)
    zeros = np.zeros(shape)
    return LiquidDiag(
        mean_mass=zeros,
        length=length.reshape(shape),
        a_len=zeros,
        c_len=zeros,
        density=np.ones(shape),
        terminal_velocity=vtm.reshape(shape),
        capacitance=zeros,
        ventilation_fv=np.ones(shape),
        ventilation_fh=np.ones(shape),
        ventilation_fkn=np.ones(shape),
        vapdep_coef1=zeros,
        vapdep_coef2=zeros,
        nre=zeros,
    )


def _make_fixed_thermo_state(t_k: float) -> ThermoState:
    """A `ThermoState` with only `tv` populated -- `coalescence_efficiency`
    reads nothing else from `thermo` (its own docstring: "thermo supplies
    the column temperature feeding thermo_fn.sfc_tension")."""
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    idx = list(ThermoState.PROPS).index(ThermoProp.tv)
    values[idx, 0, 0, 0] = t_k
    return ThermoState(values=values)


def liquid_len_vtm(nrbin: int, nbin_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin `(binb, length, terminal_velocity)` for the fixed
    `cal_breakfragment` liquid grid/air-state (H3 §8.3's "net port
    shape"): `binb` (g, `(nrbin+1,)`), `length` (cm, `(nrbin,)`),
    `terminal_velocity` (cm/s, `(nrbin,)`)."""
    grid = bin_grid.make_bin_grid("liquid", nrbin, nbin_h=nbin_h)
    binb = grid.binb

    mean_mass = _liquid_mean_mass(binb, nrbin)
    coedpi6 = float(AmpsConst.coedpi6)
    length = np.where(mean_mass > 0.0, (mean_mass / coedpi6) ** (1.0 / 3.0), 0.0)

    air = _fixed_air_state()
    vtm = _terminal_velocity(
        length, air.p_cgs, air.t_k, den_a=air.den_a, d_vis=air.d_vis, sig_wa=air.sig_wa
    )
    return binb, length, vtm


# ---------------------------------------------------------------------------
# Section 0/7: cal_breakfragment -- the driver, VERBATIM (H3 §0, §7,
# mod_amps_lib.F90:1831-2017).
# ---------------------------------------------------------------------------


def make_breakup_fragment_tables(nrbin: int, nbin_h: int) -> lookup_tables.BreakupFragmentTables:
    """`cal_breakfragment`, VERBATIM (H3 §0/§7): fixed air state (module
    docstring), the bin-pair loop `imin_bk<=i<=imax_bk=nrbin`,
    `jmin_bk<=j<=i-1` calling `coalescence_efficiency` (REUSED from
    `core.collision_kernel`, token==1 branch, H3 §3) then
    `cal_breakup_dis_ll` (H3 §2) per pair, filling `bu_fd`/`bu_tmass`.

    `jmin_bk` ("bin that has the minimum size for possible breakup", H3
    §7) is DERIVED here (the first liquid bin index with `length>=D_0`),
    not a caller-supplied argument (unlike the M1 placeholder's
    `make_breakup_fragment_tables(nrbin, jmin_bk, ...)` signature) -- the
    real `%len` computation (`liquid_len_vtm` above) now makes this
    possible; `imin_bk=jmin_bk+1`, `imax_bk=nrbin`, `jmax_bk=nrbin-1`
    follow the same formulas `lookup_tables.breakup_fragment_table_sizes`
    already encodes.

    Args:
        nrbin: total liquid bin count (cloudlab: `AmpsConfig.cloudlab().
            num_h_bins[0]` == 40).
        nbin_h: haze-split bin count (cloudlab: `AmpsConfig.cloudlab().
            nbin_h` == 20) -- see `core.bin_grid.make_bin_grid`'s own
            docstring for why this has no compile-time default.

    Returns:
        `BreakupFragmentTables(bu_tmass, bu_fd, is_placeholder=False,
        jmin_bk, imin_bk, imax_bk, jmax_bk)` -- REAL physics output, not a
        zero-filled placeholder.

    Raises:
        ValueError: if no liquid bin reaches the `D_0=0.01cm` breakup
            cutoff (would leave `jmin_bk` undefined, matching the
            Fortran's own undefined-`jmin_bk` failure mode if its
            `do i=1,NRBIN` search loop never finds a match -- raised
            explicitly here instead).
    """
    binb, length, vtm = liquid_len_vtm(nrbin, nbin_h)

    clears_d0 = length >= _D_0_CM
    if not np.any(clears_d0):
        raise ValueError(
            f"no liquid bin (of {nrbin}) reaches the D_0={_D_0_CM}cm breakup cutoff -- "
            "cannot determine jmin_bk; check the bin grid / mean-mass computation"
        )
    jmin_bk = int(np.argmax(clears_d0)) + 1  # 1-based Fortran index
    imin_bk = jmin_bk + 1
    imax_bk = nrbin
    jmax_bk = nrbin - 1

    i1d_pair_max, kk_max = lookup_tables.breakup_fragment_table_sizes(nrbin, jmin_bk)
    bu_tmass = np.zeros(i1d_pair_max, dtype=np.float64)
    bu_fd = np.zeros((2, kk_max), dtype=np.float64)

    diag = _make_liquid_diag_for_coalescence(length, vtm)
    thermo = _make_fixed_thermo_state(_T_FIXED_K)
    e_coal, cke, d_l, d_s, s_t, s_c = collision_kernel.coalescence_efficiency(diag, diag, thermo)
    del e_coal  # driver only gates on CKE (H3 §0), matching the Fortran

    for i in range(imin_bk, imax_bk + 1):
        for j in range(jmin_bk, i):
            cke_ij = float(cke[i - 1, j - 1, 0])
            if cke_ij <= 1.0e-20:
                continue
            result = cal_breakup_dis_ll(
                binb,
                nrbin,
                float(d_l[i - 1, j - 1, 0]),
                float(d_s[i - 1, j - 1, 0]),
                float(s_t[i - 1, j - 1, 0]),
                float(s_c[i - 1, j - 1, 0]),
                cke_ij,
            )
            if result is None:
                continue
            m_coal, frag_mass, frag_con = result

            i1d_pair = (j - jmin_bk + 1) + (i - imin_bk) * (1 + i - imin_bk) // 2
            bu_tmass[i1d_pair - 1] = m_coal
            kk0 = (i1d_pair - 1) * nrbin
            bu_fd[0, kk0 : kk0 + nrbin] = frag_mass
            bu_fd[1, kk0 : kk0 + nrbin] = frag_con

    return lookup_tables.BreakupFragmentTables(
        bu_tmass=bu_tmass,
        bu_fd=bu_fd,
        is_placeholder=False,
        jmin_bk=jmin_bk,
        imin_bk=imin_bk,
        imax_bk=imax_bk,
        jmax_bk=jmax_bk,
    )

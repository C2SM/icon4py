# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Collision efficiency (the `drpdrp` bilinear lookup-table gather),
coalescence efficiency, and the stochastic-collection kernel -- RAIN-RAIN
(liquid-liquid, `token 1-1`) warm case, transcribed VERBATIM from AMPS
Fortran (scale_amps repo) per
docs/superpowers/facts/m2b/coalescence-engine.md ("H1" below, the gap-fill
supplement) and docs/superpowers/facts/m2/coalescence.md ("G4" below, the
base extraction H1 supplements). M2b Task 2.

Three functions, each a thin, independently-testable building block; the
full `coalescence()` bin-pair accounting loop (`N_col`, `used_N_2`/
`used_M_2`, the `collector_loop1` scatter, `iter_loop1` over-depletion fix,
etc. -- G4 SS1, H1 SS0/SS1) is OUT OF SCOPE here, left for the M2b Task 3
engine, which is expected to call these three functions per bin-pair/
column:

* `collision_efficiency(diag_i, diag_j, luts) -> E_c` -- H1 SS3
  (`docs/superpowers/facts/m2b/coalescence-engine.md:65-119`,
  `mod_amps_core.F90:16066-16107`): the `drpdrp` clamped-bilinear gather.
  `diag_i` is group 1 ("i", the COLLECTOR -- only ITS `nre` is read,
  matching `NreL_p=max(g_1%MS(i,n)%Nre,1e-10)`); `diag_j` is group 2 ("j",
  the collectee).
* `coalescence_efficiency(diag_i, diag_j, thermo) -> (E_coal, CKE, D_L,
  D_S, S_T, S_C)` -- `cal_Coalescence_Efficiency`'s liquid-liquid branch,
  H1 SS2.1 (`mod_amps_core.F90:11842-11880`).
* `collision_kernel(diag_i, diag_j, con_j, dt, luts, *, col_level=1) -> KC`
  -- H1's kernel assembly, `KC=E_c*(vtm_i-vtm_j)*A_c*con_j*dt`
  (`docs/superpowers/facts/m2b/coalescence-engine.md:121-136`,
  `mod_amps_core.F90:16294-16310`), with `A_c` the warm rain-rain sweep
  cross-section (H1 SS2, `mod_amps_core.F90:15913-15942`). Signature note:
  the task Deliverable names this `collision_kernel(diag_i, diag_j,
  con_j, dt)` -- `luts` is an unavoidable addition (`A_c` needs no LUT,
  but `E_c` does, via `collision_efficiency`, which this function calls
  internally so callers don't have to thread `E_c` through by hand);
  `col_level` is the Fortran `col_level` branch controlling `A_c`,
  defaulted to `1` (the unconditional full-cross-section branch, matching
  `AmpsConfig.cloudlab().coll_level == 1` -- cloudlab never exercises
  `col_level==0`).

`coalescence_efficiency` does NOT feed `collision_kernel`'s `KC` (H1's own
kernel-assembly line has no `E_coal` term -- confirmed by
`mod_amps_core.F90:16294-16298` quoted in H1 SS2, `docs/superpowers/facts/
m2/coalescence.md:601-612`); `E_coal` instead feeds the SEPARATE
`used_N_2`/`used_M_2`/breakup accounting the Task 3 engine will implement
(G4 SS1.2-1.3). It is exposed here as its own function because H1's ground
truth quotes it as a standalone, reusable subroutine
(`cal_Coalescence_Efficiency`) with no dependency on the kernel/efficiency
above.

PER-VOLUME basis (H1 SS5, the M2a-lesson-driven item G4 omitted): AMPS
coalescence operates entirely on per-VOLUME quantities -- `gr%MS(i,n)%con`
is `q_state * den` (mixing ratio times air density), NOT a per-mass mixing
ratio; `class_Group.F90`'s entry/exit seam does the `*den`/`/den`
round-trip, and "no stray `/den` or `*den` appears anywhere inside
`coalescence`, `cal_collision_kernel_func`, `cal_Coalescence_Efficiency`,
or the scatter helpers" (H1 SS5). Consequently: `collision_kernel`'s
`con_j` argument MUST be a per-volume number density (bin `j`'s
`q_state*den`), NOT a mixing ratio -- the M2b Task 3 engine is responsible
for the `*den`/`/den` conversion at its own state-in/state-out boundary,
mirroring `class_Group.F90:756-758`/`:3910-3921`. This module itself
contains no `den` factor anywhere, matching the Fortran exactly.

Two distinct unit conventions coexist in this module, BOTH verbatim
transcriptions -- do not "fix" either to match the other:

* `collision_efficiency`/`collision_kernel`: CGS throughout (length cm,
  velocity cm/s, per-volume number density cm^-3), matching the rest of
  this port (`state.py`'s UNIT CONTRACT, `core/constants.py`'s
  `AmpsConst`).
* `coalescence_efficiency`: `cal_Coalescence_Efficiency` itself converts
  its CGS inputs to SI INTERNALLY (`*1.0e-2` length: cm->m; `*1.0e-3`
  surface tension: dyn/cm->N/m) and uses its OWN LOCAL SI constant
  `den_w=1000.0` (kg/m^3 -- a Fortran `real(PS),parameter` declared
  INSIDE `cal_Coalescence_Efficiency` itself, `:651`, distinct from
  `AmpsConst.den_w=1.0` g/cm^3 CGS used everywhere else in this port).
  Consequently `D_L`/`D_S` (m) and `S_T`/`S_C`/`CKE` (J) come back
  SI-valued, NOT CGS, despite `diag_i`/`diag_j` themselves being CGS --
  this is a genuine quirk of the source (a literature-calibrated formula,
  Low & List 1982-style, with SI-scaled constants `a=0.778`,
  `b=2.61e6`), kept verbatim per this task's ground-truth-first mandate
  rather than "corrected" to CGS. Only `E_coal` (dimensionless) crosses
  back into the rest of the CGS-based engine unchanged.

All functions are pure numpy, float64, array-in/array-out, operating on
`LiquidDiag` (`core/liquid_diag.py`) bundles broadcast to the bin-PAIR
shape `(nbins_i, nbins_j, npoints)`: axis 0 is group 1 ("i", the
collector, `g_1` in the Fortran), axis 1 is group 2 ("j", the collectee,
`g_2`), axis 2 is the column/point dimension every `LiquidDiag` field
already carries.
"""

from __future__ import annotations

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import thermo as thermo_fn
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    ColLutAux,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import ThermoProp, ThermoState


# ---------------------------------------------------------------------------
# Shared bin-pair broadcasting helper.
# ---------------------------------------------------------------------------


def _pairwise(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast two `(nbins, npoints)` per-bin arrays to the
    `(nbins_i, nbins_j, npoints)` bin-PAIR shape every function in this
    module operates on: `a` (group 1/"i"/collector) varies along axis 0,
    `b` (group 2/"j"/collectee) along axis 1."""
    a_b, b_b = np.broadcast_arrays(a[:, None, :], b[None, :, :])
    return a_b, b_b


# ---------------------------------------------------------------------------
# collision_efficiency -- the drpdrp bilinear gather, H1 SS3.
# ---------------------------------------------------------------------------

# ec_min, H1 SS2 declaration (`real(PS),parameter :: ec_min=0.0D0`, `:15887`).
_EC_MIN = 0.0


def _clamp_efficiency(raw: np.ndarray) -> np.ndarray:
    """The 2-stage `E_c` clamp, H1 SS3 verbatim (`mod_amps_core.F90:
    16099-16107`): `E_c=min(1.0,max(ec_min,raw))`, then a SEPARATE
    `if(E_c>15.0) E_c=15.0`. The second clamp is UNREACHABLE through the
    first (`min(1.0,...)` already caps `E_c<=1.0<15.0`) -- kept anyway
    for verbatim fidelity; `test_collision_kernel.py`'s
    `test_clamp_helper_15_branch_is_unreachable_dead_code` exercises this
    helper directly and documents why."""
    e_c = np.minimum(1.0, np.maximum(_EC_MIN, raw))
    return np.where(e_c > 15.0, 15.0, e_c)


def _bilinear_gather(x: np.ndarray, y: np.ndarray, table: np.ndarray, aux: ColLutAux) -> np.ndarray:
    """Generic clamped bilinear gather over a `col_lut_aux`-described 2D
    table -- H1 SS3's canonical formula (`docs/superpowers/facts/m2/
    coalescence.md:979-980`): `x` indexes the table's SECOND (column,
    `%nc`/`%xs`/`%dx`) axis, `y` its FIRST (row, `%nr`/`%ys`/`%dy`) axis;
    both index formulas use Fortran `int()` (truncate-toward-zero,
    `np.trunc` here -- see the note below) then clamp to `[1, n-1]`
    (1-based) before a standard 4-point bilinear interpolation. Returns
    the RAW interpolated value -- callers apply their own final clamp
    (`_clamp_efficiency` for `drpdrp`; H1 SS3 notes the other 6 ice
    tables share this identical pattern with different axis inputs, out
    of this module's rain-rain scope).

    `np.trunc` vs `np.floor`: only diverges from Fortran's `int()` for a
    NEGATIVE `(coord-origin)/step`, i.e. `coord` below the table's
    origin -- exactly the case where both `+1` results, before clamping,
    are `<=0` and so clamp to the SAME minimum index `1` regardless (for
    non-negative `(coord-origin)/step`, `trunc` and `floor` are
    identical). `np.trunc` is used anyway to match `int()` literally
    rather than relying on that post-clamp equivalence argument.
    """
    j1 = np.clip(np.trunc((x - aux.xs) / aux.dx).astype(np.int64) + 1, 1, aux.nc - 1)
    x1 = (j1 - 1).astype(np.float64) * aux.dx + aux.xs

    i1 = np.clip(np.trunc((y - aux.ys) / aux.dy).astype(np.int64) + 1, 1, aux.nr - 1)
    y1 = (i1 - 1).astype(np.float64) * aux.dy + aux.ys

    wx = np.clip((x - x1) / aux.dx, 0.0, 1.0)
    wy = np.clip((y - y1) / aux.dy, 0.0, 1.0)

    i0, j0 = i1 - 1, j1 - 1  # Fortran 1-based -> numpy 0-based
    t00 = table[i0, j0]
    t01 = table[i0, j0 + 1]
    t10 = table[i0 + 1, j0]
    t11 = table[i0 + 1, j0 + 1]

    return (
        (1.0 - wx) * (1.0 - wy) * t00
        + (1.0 - wx) * wy * t01
        + wx * (1.0 - wy) * t10
        + wx * wy * t11
    )


def collision_efficiency(diag_i: LiquidDiag, diag_j: LiquidDiag, luts: AmpsLuts) -> np.ndarray:
    """Warm rain-rain `drpdrp` collision-efficiency gather, H1 SS3
    verbatim (`docs/superpowers/facts/m2b/coalescence-engine.md:65-119`,
    `mod_amps_core.F90:16066-16107`):

        NreL_p = max(g_1%MS(i,n)%Nre, 1e-10)
        rrat = g_2%MS(j,n)%len / g_1%MS(i,n)%len
        if rrat < 0: rrat_p = 0.0                  # degenerate guard
        elif rrat > 1: rrat_p = 1.0 / rrat          # fold to <=1
        else: rrat_p = rrat
        E_c = clamp(bilinear_gather(rrat_p, log10(NreL_p), drpdrp), ec_min, 1, 15)

    `diag_i` is group 1 ("i", the COLLECTOR) -- only its `nre` is read
    (`NreL_p` uses `g_1`'s `Nre` exclusively, never `g_2`'s); `diag_j` is
    group 2 ("j", the collectee) -- only its `length` is read (via
    `rrat`). This port's own addition, not in the Fortran: `rrat`'s
    division-by-zero (`diag_i.length<=0`, an inactive/degenerate collector
    bin) is routed into the SAME `rrat<0` degenerate branch (`rrat_p=0`)
    rather than propagating a NaN/Inf `0/0` or `x/0` -- a documented,
    physically-motivated substitute for an undefined Fortran edge case.

    `em` (the Fortran's own debug marker, set but only ever read by a
    `write(*,*)` under `debug`) is NOT modeled -- it has no effect on
    `E_c` itself.

    Returns:
        `E_c`, shape `(nbins_i, nbins_j, npoints)`, `float64`, clamped to
        `[ec_min, 1.0]` (the `>15` stage is unreachable, see
        `_clamp_efficiency`).
    """
    len_i, len_j = _pairwise(diag_i.length, diag_j.length)
    nre_i, _nre_j = _pairwise(diag_i.nre, diag_j.nre)  # only group-1 ("i")'s Nre is used

    with np.errstate(divide="ignore", invalid="ignore"):
        rrat = np.where(len_i > 0.0, len_j / np.where(len_i > 0.0, len_i, 1.0), -1.0)

    nre_l_p = np.maximum(nre_i, 1.0e-10)  # NreL_p; unaffected by the rrat<0 branch (see H1 SS3)

    negative = rrat < 0.0
    reciprocal = rrat > 1.0
    rrat_p = np.where(negative, 0.0, np.where(reciprocal, 1.0 / rrat, rrat))

    raw = _bilinear_gather(rrat_p, np.log10(nre_l_p), luts.drpdrp, luts.adrpdrp)
    return _clamp_efficiency(raw)


# ---------------------------------------------------------------------------
# coalescence_efficiency -- cal_Coalescence_Efficiency, liquid-liquid
# (token 1-1) branch, H1 SS2.1 verbatim (`mod_amps_core.F90:11842-11880`).
# ---------------------------------------------------------------------------

_A_COEF = 0.778  # H1 SS2.1 `a` (`:653`)
_B_COEF = 2.61e6  # H1 SS2.1 `b` (`:653`)
_D_0_CM = 0.01  # H1 SS2.1 `D_0` (`:654`), cm -- compared against the SI D_L/D_S*100 cm round-trip
# H1 SS2.1's OWN LOCAL `real(PS),parameter :: den_w = 1000.0D0` (`:651`), kg/m^3 SI -- distinct
# from AmpsConst.den_w (1.0 g/cm^3 CGS) used by every other CGS formula in this port.
_DEN_W_SI = 1000.0


def coalescence_efficiency(
    diag_i: LiquidDiag, diag_j: LiquidDiag, thermo: ThermoState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`cal_Coalescence_Efficiency`, liquid-liquid (`token 1-1`) branch,
    H1 SS2.1 verbatim (`mod_amps_core.F90:11842-11880`). Returns
    `(E_coal, CKE, D_L, D_S, S_T, S_C)`, matching this task's Deliverable
    order; `dS_S` (`S_T-S_C`, also an `intent(out)` in the Fortran) is
    computed internally but not part of this return tuple.

    UNIT QUIRK, kept verbatim (see module docstring): `D_L`/`D_S` come
    back in METERS, `S_T`/`S_C`/`CKE` in JOULES -- SI, not CGS, despite
    `diag_i`/`diag_j` being CGS (cm, cm/s). Only `E_coal` (dimensionless)
    is CGS-agnostic.

    `diag_i`/`diag_j` need only `length` (cm) and `terminal_velocity`
    (cm/s) here; `thermo` supplies the column temperature feeding
    `thermo_fn.sfc_tension` (`th_var%sig_wa`, CGS dyn/cm == erg/cm^2).

    For bin pairs below the `D_0` size cutoff
    (`min(D_L,D_S)*100 < D_0`), the Fortran sets `E_coal=1.0` and leaves
    `CKE`/`S_T`/`S_C`/`dS_S` UNINITIALIZED (an early `if/else` branch that
    never assigns them -- genuine Fortran garbage, never read downstream
    since a fully-coalescing pair has no further use for them: `E_coal=1`
    is the only value any real caller reads in that branch, and IS
    verbatim). This port instead computes `CKE`/`S_T`/`S_C` UNCONDITIONALLY
    (well-defined given `D_L>0`, matching what the formula WOULD give) --
    a documented, physically-motivated substitute for propagating
    undefined values, not a deviation in anything a real caller reads.

    Returns:
        `(E_coal, CKE, D_L, D_S, S_T, S_C)`, each shape
        `(nbins_i, nbins_j, npoints)`, `float64`.
    """
    len_i, len_j = _pairwise(diag_i.length, diag_j.length)  # cm
    vtm_i, vtm_j = _pairwise(diag_i.terminal_velocity, diag_j.terminal_velocity)  # cm/s

    t = get_thermo_prop(thermo, ThermoProp.tv)  # K, (npoints,)
    sig_wa = thermo_fn.sfc_tension(t)  # erg/cm^2 == dyn/cm, CGS, (npoints,)
    sig_wa_b = np.broadcast_to(sig_wa[None, None, :], len_i.shape)

    d_l = np.maximum(len_i, len_j) * 1.0e-2  # m
    d_s = np.minimum(len_i, len_j) * 1.0e-2  # m
    below_d0 = np.minimum(d_l, d_s) * 100.0 < _D_0_CM

    v_l = np.maximum(vtm_i, vtm_j) * 1.0e-2  # m/s
    v_s = np.minimum(vtm_i, vtm_j) * 1.0e-2  # m/s
    sig_si = sig_wa_b * 1.0e-3  # N/m

    pi = float(AmpsConst.PI)
    s_t = pi * sig_si * (d_l**2.0 + d_s**2.0)
    s_c = pi * sig_si * (d_l**3.0 + d_s**3.0) ** (2.0 / 3.0)
    ds_s = s_t - s_c

    with np.errstate(divide="ignore", invalid="ignore"):
        cube_sum = d_l**3.0 + d_s**3.0
        cke = (
            (_DEN_W_SI * pi / 12.0)
            * (v_l - v_s) ** 2.0
            * (d_l * d_s) ** 3.0
            / np.where(cube_sum > 0.0, cube_sum, 1.0)
        )
    e_t = cke + ds_s

    small_et = e_t < 5.0e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        d_s_over_d_l = np.where(d_l > 0.0, d_s / np.where(d_l > 0.0, d_l, 1.0), 0.0)
        exponent = -_B_COEF * sig_si * e_t**2.0 / np.where(s_c > 0.0, s_c, 1.0)
        e_coal_main = np.where(
            small_et,
            _A_COEF * (1.0 + d_s_over_d_l) ** (-2.0) * np.exp(exponent),
            0.0,
        )

    e_coal = np.where(below_d0, 1.0, e_coal_main)

    return e_coal, cke, d_l, d_s, s_t, s_c


# ---------------------------------------------------------------------------
# collision_kernel -- KC = E_c*(vtm_i-vtm_j)*A_c*con_j*dt, H1 kernel
# assembly + warm rain-rain sweep cross-section, H1 SS2.
# ---------------------------------------------------------------------------


def collision_kernel(
    diag_i: LiquidDiag,
    diag_j: LiquidDiag,
    con_j: np.ndarray,
    dt: float,
    luts: AmpsLuts,
    *,
    col_level: int = 1,
) -> np.ndarray:
    """`KC = E_c*(vtm_i-vtm_j)*A_c*con_j*dt` -- H1's kernel assembly
    (`docs/superpowers/facts/m2b/coalescence-engine.md:121-136`,
    `mod_amps_core.F90:16294-16310`), with `A_c` the warm rain-rain
    (`token 1-1`) sweep cross-section (H1 SS2, `mod_amps_core.F90:
    15913-15942`):

        A_c = 0 if col_level==0 and (len_i>1e-4 or len_j>1e-4) else
              0.25*pi*(len_i+len_j)**2

    PER-VOLUME: `con_j` must be a per-volume number density
    (`gr%MS(j,n)%con = q_state*den`), NOT a mixing ratio -- see module
    docstring's PER-VOLUME basis note; this function (like the Fortran)
    contains no `den` factor.

    Signature note: the task Deliverable names this
    `collision_kernel(diag_i, diag_j, con_j, dt)` -- `luts` is an
    unavoidable addition (`E_c`, via `collision_efficiency`, needs the
    `drpdrp` LUT); `col_level` is the Fortran `col_level` branch above,
    defaulted to `1` (the unconditional full-cross-section branch,
    matching `AmpsConfig.cloudlab().coll_level == 1` -- cloudlab never
    exercises `col_level==0`, whose own `>1e-4 cm` (1 micron) condition
    is true for essentially any realistic rain/cloud bin and so would
    zero `A_c` almost everywhere; kept verbatim regardless).

    Args:
        diag_i: group 1 ("i", collector) `LiquidDiag`.
        diag_j: group 2 ("j", collectee) `LiquidDiag`.
        con_j: per-volume number density of group 2's bins, shape
            `(nbins_j, npoints)`, matching `diag_j`'s own per-bin shape.
        dt: collision-substep timestep (s), scalar.
        luts: `AmpsLuts` (for `collision_efficiency`'s `drpdrp` gather).
        col_level: the Fortran `col_level` sweep-cross-section branch
            selector (see above); default `1`.

    Returns:
        `KC`, shape `(nbins_i, nbins_j, npoints)`, `float64`.
    """
    e_c = collision_efficiency(diag_i, diag_j, luts)

    len_i, len_j = _pairwise(diag_i.length, diag_j.length)
    a_c_full = 0.25 * float(AmpsConst.PI) * (len_i + len_j) ** 2.0
    if col_level == 0:
        a_c = np.where((len_j > 1.0e-4) | (len_i > 1.0e-4), 0.0, a_c_full)
    else:
        a_c = a_c_full

    vtm_i, vtm_j = _pairwise(diag_i.terminal_velocity, diag_j.terminal_velocity)

    con_j = np.asarray(con_j, dtype=np.float64)
    con_j_b = np.broadcast_to(con_j[None, :, :], e_c.shape)

    return e_c * (vtm_i - vtm_j) * a_c * con_j_b * dt

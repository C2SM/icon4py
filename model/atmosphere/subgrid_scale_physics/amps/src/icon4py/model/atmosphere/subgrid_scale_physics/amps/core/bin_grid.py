# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Bin boundary construction, transcribed verbatim from AMPS Fortran
(scale_amps repo) per docs/superpowers/facts/m1/bins-config-indexmaps.md
("F2" in docstrings below), §1 (`binmicrosetup_scale`,
mod_amps_lib.F90:328-618).

Covers the liquid (rain+haze, 40/80-bin) and ice (10/20/40-bin) bin-boundary
constructions and their `binb(i+1) = a_b*binb(i) + b_b` recurrence. No
aerosol-bin (`binba`) construction is shown anywhere in F2 §1 -- only its
downstream consumer declares a bare `binba(82)` array (F2 §6c,
class_Cloud_Micro.F90 dummy argument) with no construction code quoted --
so `make_bin_grid` supports only `token in {"liquid", "ice"}`.

`a_b`/`b_b` here are PER-BIN arrays, not the single scalars of the general
`class_Group` abstraction (F2 §7: `g%a_b = dab; g%b_b = dbb`, a single
ratio/additive-constant pair). That single-scalar recurrence
(`g%binb(i) = a_b*g%binb(i-1) + b_b`) is dead code in the actual call path
for liquid/ice groups -- `make_Group` is always called with the boundary
array `dbinb` already precomputed by `binmicrosetup_scale` and copied
verbatim (F2 §7 lines 1310-1313); the commented-out `!tmp` fallback that
would use a single scalar `a_b`/`b_b` to build boundaries from scratch is
never reached. `binmicrosetup_scale` itself constructs liquid bins from TWO
segments (haze, then main) and ice bins from THREE segments (mg1->mg2,
mg2->mg3, mg3->mg4), each with its own ratio -- so representing the
recurrence faithfully requires a per-bin coefficient, not one scalar pair.

NEEDS_CONTEXT (flagged per the task's ground-truth rule, not guessed):

* `nbin_h` (the haze-split bin count) has NO value anywhere in F2. F2 §1's
  own supplementary block states it explicitly: "nbin_h has no
  compile-time default -- it is set only via namelist
  `PARAM_ATMOS_PHY_MP_AMPS_bin`" (F2 §1, com_amps.F90 declaration note).
  F2 §6b's own namelist-field comment ("e.x. 20 for 40 liq. bins") is
  explicitly an illustrative example, not a confirmed cloudlab run value.
  Consequently `make_bin_grid` takes `nbin_h` as a REQUIRED keyword
  argument for `token="liquid"` (raises ValueError if omitted) rather than
  hardcoding a guessed value; the actual cloudlab-run value is left for the
  `AmpsConfig`/run.conf-driven task to supply (see the M1 plan's Task 4
  brief, which explicitly owns `PARAM_ATMOS_PHY_MP_AMPS_bin` fields
  including `nbin_h`).
* `minmass_s` (the ice-bin seed) is a namelist-read variable with no F2 §1
  value either (only declared, undefined, in the same supplementary
  block). However F2 §5 (`AMPSTASK.F`, the actual cloudlab task file)
  DOES quote its real cloudlab value: `minmass_s = 4.18879020478639E-12`
  (F2 §5 line 754) -- bit-for-bit equal to F2 §1's `mg1` PARAMETER. This is
  used below as `ICE_MINMASS_CLOUDLAB`, the default for the `minmass_ice`
  keyword argument (still overridable, not hardcoded into the recurrence
  itself).
* "mean masses" (mentioned in the task's Interfaces bullet as a desired
  `BinGrid` field) has no formula anywhere in F2 -- neither §1 nor
  elsewhere in the fact file. Rather than invent a bin-center convention
  (e.g. geometric mean of adjacent boundaries) with no F2 grounding, this
  field is DELIBERATELY OMITTED from `BinGrid`; only `binb` (the F2-quoted
  boundary array) is exposed. See the task report for this call-out.
* `isplit_bin_liq`/`isplit_bin_ice` (F2 §1 lines 281-297, "define split
  bin") is a DIFFERENT concept from the haze split (it locates where
  liquid and ice bins correspond by physical size, ~20 micron radius
  threshold) and needs BOTH a liquid and an ice grid simultaneously, which
  does not fit this module's per-token `make_bin_grid(token, nbins)`
  interface. Left out of scope for this task; `BinGrid.nbin_h` is the
  "haze-split index" the task's ground-truth pointer names ("haze split
  nbin_h").

Fortran-exactness note on the liquid last boundary: F2 §1 sets
`binbr(nbr+1) = maxmass_r` BEFORE the main-segment do-loop, but that
loop's own upper bound is `nbr+1` (`do i=nbin_h+2,nbr+1`), so its LAST
iteration overwrites the direct assignment with the loop-computed value
`binbr(nbr)*dsrat`. The final boundary is therefore only asymptotically
(to floating-point rounding) equal to `maxmass_r`, not bit-exact --
preserved here by looping through index `nbins` inclusive rather than
special-casing it. The haze/main segment boundary (`binbr(nbin_h+1) =
c_max`) has no such overwrite (neither loop's range includes that index a
second time) and IS bit-exact to the `C_MAX` literal.
"""

from __future__ import annotations

import dataclasses

import numpy as np


# ---------------------------------------------------------------------------
# Seed constants: true Fortran PARAMETER literals from F2 §1
# (binmicrosetup_scale), always the same regardless of namelist config.
# ---------------------------------------------------------------------------

C_MNM = 4.188790205e-15  # minimum cloud droplet mass (g); F2 §1 line 63
C_MAX = 6.54498e-8  # haze/main segment boundary (g); F2 §1 line 63
MAXMASS_R = 5.2359870e-01  # 1 cm rain-drop diameter (g); F2 §1 line 70

# Mass boundaries defining pristine crystal / aggregate / hail categories
# for the ice bin construction; F2 §1 lines 76-77 (active, uncommented
# PARAMETER statement -- NOT the commented-out `!ccc` alternates above it).
MG1 = 4.18879020478639e-12
MG2 = 1.0e-6
MG3 = 1.0e-2
MG4 = 1.0e1

# Per-segment bin counts for the ice construction, keyed by total ice bins;
# F2 §1 lines 86-88 (active DATA values, not the commented `!ccc` ones).
NBIN20 = (4, 10, 6)
NBIN10 = (2, 5, 3)
NBIN40 = (8, 20, 12)

LIQUID_NBINS = (40, 80)  # F2 §1 lines 113-116 (PRC_abort otherwise)
ICE_NBINS = (10, 20, 40)  # F2 §1 lines 117-120, 258-260 (PRC_abort otherwise)

# F2 §5 (AMPSTASK.F, cloudlab run config): minmass_s = 4.18879020478639E-12
# -- see the module docstring's NEEDS_CONTEXT note. Equals MG1 bit-for-bit.
ICE_MINMASS_CLOUDLAB = 4.18879020478639e-12


@dataclasses.dataclass(frozen=True)
class BinGrid:
    """Bin boundaries and per-bin recurrence coefficients for one
    hydrometeor category (liquid or ice), F2 §1.

    Attributes:
        token: "liquid" or "ice".
        nbins: number of bins (LIQUID_NBINS or ICE_NBINS, validated).
        binb: bin boundaries (mass, g), length `nbins + 1`.
        a_b: per-bin multiplicative recurrence coefficient, length
            `nbins`; `binb[i+1] == a_b[i]*binb[i] + b_b[i]`.
        b_b: per-bin additive recurrence coefficient, length `nbins`;
            always zero (no F2 §1 reachable branch uses an additive term
            for liquid/ice -- see module docstring).
        nbin_h: the haze-split bin count for liquid grids (F2 §1's
            `nbin_h`); `None` for ice grids (haze splitting is a
            liquid/rain-only concept in F2 §1).
    """

    token: str
    nbins: int
    binb: np.ndarray
    a_b: np.ndarray
    b_b: np.ndarray
    nbin_h: int | None


def make_bin_grid(
    token: str,
    nbins: int,
    *,
    nbin_h: int | None = None,
    minmass_ice: float = ICE_MINMASS_CLOUDLAB,
) -> BinGrid:
    """Construct bin boundaries for one hydrometeor category, F2 §1
    (`binmicrosetup_scale`).

    Args:
        token: "liquid" (rain + haze, F2 §1 lines 169-204) or "ice"
            (F2 §1 lines 216-265).
        nbins: total bin count; validated against LIQUID_NBINS (40, 80)
            or ICE_NBINS (10, 20, 40) per `token`.
        nbin_h: haze-split bin count, REQUIRED for `token="liquid"`. F2
            gives this no value (namelist-only, no compile-time default;
            see module docstring) -- callers must supply the actual
            cloudlab-run value explicitly rather than have one guessed
            here. Must satisfy `1 <= nbin_h < nbins`. Ignored for
            `token="ice"`.
        minmass_ice: ice-bin seed mass (g), only used for `token="ice"`.
            Defaults to `ICE_MINMASS_CLOUDLAB`, F2 §5's quoted cloudlab
            AMPSTASK.F value for `minmass_s` (see module docstring).

    Returns:
        A frozen `BinGrid`.

    Raises:
        ValueError: unknown `token`; `nbins` not in the allowed set for
            `token`; `token="liquid"` with `nbin_h` missing or out of
            `[1, nbins)`.
    """
    if token == "liquid":
        return _make_liquid_bin_grid(nbins, nbin_h)
    elif token == "ice":
        return _make_ice_bin_grid(nbins, minmass_ice)
    else:
        raise ValueError(f"token must be 'liquid' or 'ice', got {token!r}")


def _make_liquid_bin_grid(nbins: int, nbin_h: int | None) -> BinGrid:
    if nbins not in LIQUID_NBINS:
        raise ValueError(f"# of bins for liquid is not 40 or 80: {nbins}")
    if nbin_h is None:
        raise ValueError(
            "nbin_h (haze-split bin count) is required for liquid bin grids: "
            "F2 gives it no compile-time default (namelist-only PARAM_ATMOS_PHY_MP_AMPS_bin "
            "field) -- see bin_grid.py module docstring NEEDS_CONTEXT note."
        )
    if not (1 <= nbin_h < nbins):
        raise ValueError(f"nbin_h must satisfy 1 <= nbin_h < nbins; got {nbin_h=}, {nbins=}")

    binb = np.empty(nbins + 1, dtype=np.float64)
    a_b = np.empty(nbins, dtype=np.float64)
    b_b = np.zeros(nbins, dtype=np.float64)

    # Haze segment: binbr(1) = c_mnm; do i=2,nbin_h: binbr(i)=binbr(i-1)*dsrat_h
    # (F2 §1 lines 170-175). binbr(nbin_h+1) = c_max is a DIRECT assignment
    # (line 177), NOT computed via the dsrat_h recurrence.
    binb[0] = C_MNM
    dsrat_h = (C_MAX / C_MNM) ** (1.0 / nbin_h)
    for i in range(1, nbin_h):
        binb[i] = binb[i - 1] * dsrat_h
    binb[nbin_h] = C_MAX
    a_b[:nbin_h] = dsrat_h

    # Main segment: dsrat=(maxmass_r/c_max)**(1/(nbr-nbin_h)) (line 184);
    # do i=nbin_h+2,nbr+1: binbr(i)=binbr(i-1)*dsrat (lines 200-204) --
    # this loop's own upper bound overwrites the earlier direct assignment
    # `binbr(nbr+1)=maxmass_r` (line 186) on its final iteration; see
    # module docstring's Fortran-exactness note.
    dsrat = (MAXMASS_R / C_MAX) ** (1.0 / (nbins - nbin_h))
    for i in range(nbin_h + 1, nbins + 1):
        binb[i] = binb[i - 1] * dsrat
    a_b[nbin_h:] = dsrat

    return BinGrid(token="liquid", nbins=nbins, binb=binb, a_b=a_b, b_b=b_b, nbin_h=nbin_h)


def _make_ice_bin_grid(nbins: int, minmass_ice: float) -> BinGrid:
    if nbins == 40:
        n1, n2, n3 = NBIN40
    elif nbins == 20:
        n1, n2, n3 = NBIN20
    elif nbins == 10:
        n1, n2, n3 = NBIN10
    else:
        raise ValueError(f"# of bins for ice is not 40, 20, 10: {nbins}")

    binb = np.empty(nbins + 1, dtype=np.float64)
    a_b = np.empty(nbins, dtype=np.float64)
    b_b = np.zeros(nbins, dtype=np.float64)

    # binbi(1) = minmass_s (F2 §1 line 218/232/246, per-branch but identical
    # pattern); three segments, each with its own ratio, purely
    # multiplicative throughout (no direct-assignment overwrite quirk here,
    # unlike the liquid case -- see module docstring).
    binb[0] = minmass_ice

    srat_s = (MG2 / MG1) ** (1.0 / n1)
    for i in range(1, n1 + 1):
        binb[i] = binb[i - 1] * srat_s
    a_b[:n1] = srat_s

    srat_s = (MG3 / MG2) ** (1.0 / n2)
    for i in range(n1 + 1, n1 + n2 + 1):
        binb[i] = binb[i - 1] * srat_s
    a_b[n1 : n1 + n2] = srat_s

    srat_s = (MG4 / MG3) ** (1.0 / n3)
    for i in range(n1 + n2 + 1, n1 + n2 + n3 + 1):
        binb[i] = binb[i - 1] * srat_s
    a_b[n1 + n2 :] = srat_s

    return BinGrid(token="ice", nbins=nbins, binb=binb, a_b=a_b, b_b=b_b, nbin_h=None)

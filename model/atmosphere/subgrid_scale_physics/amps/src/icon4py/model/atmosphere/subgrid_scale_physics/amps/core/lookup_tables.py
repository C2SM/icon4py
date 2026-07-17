# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""AMPS lookup-table loaders and computed-table generators, transcribed
from docs/superpowers/facts/m1/lut-files.md ("F3" in comments below).

Two kinds of tables:

* Read from disk (F3 SS1-4): the collision-efficiency LUTs, habit-frequency
  tables, diagnostic maps, `lmt_mass_*`, and `znorm`, converted offline by
  `../../codegen/convert_luts.py` (repo-relative:
  `model/atmosphere/subgrid_scale_physics/amps/codegen/convert_luts.py`)
  from the real AMPS_DATA files into two packaged npz archives,
  `data/amps_luts_collision.npz` (the 7 collision-efficiency LUTs) and
  `data/amps_luts_misc.npz` (everything else) -- split so each individually
  clears the repo's `check-added-large-files` pre-commit hook (default
  500 KB; see `data/README.md` for provenance and the size numbers that
  drove the split) -- loaded here via `importlib.resources` (F5 SS3
  packaged-data pattern) and merged transparently; `load_luts()`'s own
  signature/return type is unaffected by the two-file detail.
* Computed at load time, no data file (F3 SS5): osmotic-coefficient LUTs
  (`init_osmo_par`), the normal/inverse-normal CDF LUTs
  (`init_normal_lut`/`init_inv_normal_lut`), and the Inherent Growth
  Parameterization spline knots (`init_inherent_growth_par`) -- all pure
  constants, transcribed VERBATIM where F3 quotes exact Fortran source.

`init_osmo_par`'s x-grid NEEDS_CONTEXT (flagged in the M1 Task 5 report) is
RESOLVED: `osm_ammsul`/`osm_sodchl` were read directly in
`mod_amps_utility.F90` (~line 12988/13054, coordinator-authorized) and
transcribed verbatim below -- the real interpolation-node x-grid is
non-uniform (dense 0.1 up to molality 1.0, medium 0.2 up to 2.0, coarse 0.5
to the domain end), not the uniform grid the original (F3-only) submission
had inferred.

One exception, NOT computed as part of `load_luts()`: the Low-List
collisional-breakup fragment tables (`bu_fd`/`bu_tmass`, F3 SS5.5). F3's
quoted `cal_breakfragment` needs a runtime bin-count `NRBIN` (config-
dependent, unknown here) AND `jmin_bk` (the smallest liquid bin whose
diameter clears the `D_0=0.01cm` breakup cutoff). Per coordinator
authorization, the call chain was followed directly in the Fortran beyond
what F3 quotes: `cal_Coalescence_Efficiency` and `cal_breakup_dis_LL`
(`mod_amps_core.F90`, the fragment-distribution math itself, ~429 lines,
not further reducible) each call additional helpers
(`cal_sig_sf`/`cal_Hmusig`/`zbrent`/`getznorm2`), and BOTH need `%len`
(diameter) and `%vtm` (terminal velocity) per liquid bin, which the
Fortran computes via `ini_group_mp` -> `cal_meanmass_vec` -> `diag_pq`'s
liquid branch -> `cal_den_aclen_vec` + `cal_terminal_vel_vec`, which in
turn need a steady-state air/thermo object built by `make_AirGroup_2` ->
`Make_Thermo_Var3_2`. That is 13 additional named routines across 6 files
(mod_amps_core.F90, class_Group.F90, class_Mass_Bin.F90,
class_AirGroup.F90, class_Thermo_Var.F90, mod_amps_utility.F90) and
roughly 1200-1400 lines of Fortran once trimmed to only the liquid/
spherical branches actually exercised -- well past the ~400-line transcription
budget for this task. Per the coordinator's own scope guard ("if the
transcription balloons past ~400 lines, STOP and report BLOCKED with the
call-tree inventory instead"), the FILL is left BLOCKED (still
zero-initialized, matching the Fortran's own pre-loop state
`bu_fd=0.0_PS; bu_tmass=0.0_PS`) -- see the M1 Task 5 report's call-tree
inventory for the full routine list, each with its file:line range, as a
scoped follow-up task. Only the SIZE/allocation formula
(`breakup_fragment_table_sizes`/`make_breakup_fragment_tables`, verbatim
and exact, cross-checked against F3's own quoted 80-bin declaration) is
provided here, unchanged from the original submission. `jmin_bk` is a
REQUIRED argument to both functions -- the same "no compile-time default,
caller must supply it" pattern `bin_grid.py` already uses for `nbin_h`.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import math

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Aux/holder dataclasses mirroring the Fortran `sequence` derived types
# quoted in F3 SS4 (class_Group.F90, class_Mass_Bin.F90).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ColLutAux:
    """2D collision-efficiency LUT axis metadata, `col_lut_aux`
    (`class_Group.F90:179-183`, F3 SS4). `xs`/`dx` describe the row axis,
    `ys`/`dy` the column axis (F3 SS6: "x = row axis, y = column axis")."""

    xs: float
    dx: float
    ys: float
    dy: float
    nr: int
    nc: int


@dataclasses.dataclass(frozen=True)
class GridAux:
    """Plain `(nrow, ncol)` header for the habit-frequency/diagnostic-map/
    `lmt_mass_*` files (F3 SS3: "1 comment line + nrow ncol", no
    `xs/dx/ys/dy` -- "grid is implicit")."""

    nr: int
    nc: int


@dataclasses.dataclass(frozen=True)
class Data1DLut:
    """1D lookup table, `data1d_lut`/`data1d_lut_big`
    (`class_Mass_Bin.F90:281-295`, F3 SS4): `y[i]` sampled at
    `xs + dx*i` for `i` in `range(n)`."""

    n: int
    xs: float
    dx: float
    y: np.ndarray


@dataclasses.dataclass(frozen=True)
class VapIgpAux:
    """Inherent Growth Parameterization spline knots, `vap_igp_aux`
    (`class_Group.F90:186-195`, F3 SS4). `x` are the knot temperatures
    (deg C); `a`/`b` are per-interval cubic coefficients."""

    nok: int
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray


@dataclasses.dataclass(frozen=True)
class AmpsLuts:
    """All AMPS lookup tables: file-derived (F3 SS1-4) plus computed
    (F3 SS5), as returned by `load_luts()`. Does NOT include the breakup
    fragment tables -- see the module docstring."""

    # -- Collision-efficiency 2D LUTs, RDCETB (F3 SS2.1, SS4) --------------
    drpdrp: np.ndarray
    adrpdrp: ColLutAux
    hexdrp: np.ndarray
    ahexdrp: ColLutAux
    bbcdrp: np.ndarray
    abbcdrp: ColLutAux
    coldrp: np.ndarray
    acoldrp: ColLutAux
    gp1drp: np.ndarray
    agp1drp: ColLutAux
    gp4drp: np.ndarray
    agp4drp: ColLutAux
    gp8drp: np.ndarray
    agp8drp: ColLutAux

    # -- Habit-frequency tables, RDCETB (F3 SS2.1 lines 161-215) -----------
    # pol/pla/col are already clipped-to-zero and renormalized by their
    # per-cell sum; ros/ppo are clipped-to-zero only (F3 SS6).
    pol_frq: np.ndarray
    pla_frq: np.ndarray
    col_frq: np.ndarray
    ros_frq: np.ndarray
    ppo_frq: np.ndarray
    frq_aux: GridAux

    # -- Diagnostic a/c-axis + density maps, RDCETB (F3 SS2.1 lines 217-259) --
    mtac_map_col: np.ndarray  # (nr, nc, 2): [..., 0]=tmp (a/c axis), [...,1]=tmd (density)
    mtac_map_pla: np.ndarray
    map_col_aux: GridAux
    map_pla_aux: GridAux

    # -- lmt_mass_{col,pla}, RDCETB (F3 SS2.1 lines 232-259) ---------------
    lmt_mass_col: np.ndarray  # (50,)
    lmt_mass_pla: np.ndarray  # (50,)
    lmt_mass_col_aux: GridAux
    lmt_mass_pla_aux: GridAux

    # -- Standard normal distribution table, RDSTTB (F3 SS2.3) -------------
    znorm: np.ndarray  # (451, 4)

    # -- Computed tables, no data file (F3 SS5) -----------------------------
    osm_nh42so4: Data1DLut
    osm_sodchl: Data1DLut
    snrml: Data1DLut
    isnrml: Data1DLut
    vigp: VapIgpAux


# ---------------------------------------------------------------------------
# Packaged-data loading (F5 SS3 pattern).
# ---------------------------------------------------------------------------

_PACKAGE = "icon4py.model.atmosphere.subgrid_scale_physics.amps.data"
# Two npz files, not one: each individually must clear the repo's
# `check-added-large-files` pre-commit hook (default 500 KB). Split by
# codegen/convert_luts.py's `split_tables()`: `_COLLISION_NPZ_NAME` holds
# the 7 collision-efficiency LUTs (the largest tables), `_MISC_NPZ_NAME`
# holds everything else. This API (`load_luts() -> AmpsLuts`) is unchanged
# by the split -- callers never see the two-file detail.
_COLLISION_NPZ_NAME = "amps_luts_collision.npz"
_MISC_NPZ_NAME = "amps_luts_misc.npz"


def _col_lut_aux(raw: np.ndarray) -> ColLutAux:
    xs, dx, ys, dy, nr, nc = raw
    return ColLutAux(xs=float(xs), dx=float(dx), ys=float(ys), dy=float(dy), nr=int(nr), nc=int(nc))


def _grid_aux(raw: np.ndarray) -> GridAux:
    nr, nc = raw
    return GridAux(nr=int(nr), nc=int(nc))


def _load_npz(name: str) -> dict[str, np.ndarray]:
    resource = importlib.resources.files(_PACKAGE).joinpath(name)
    with importlib.resources.as_file(resource) as path, np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def load_luts() -> AmpsLuts:
    """Load the packaged AMPS_DATA-derived lookup tables (via
    `importlib.resources` on this package's `data/` directory, transparently
    merging the two split npz files -- see `_COLLISION_NPZ_NAME`/
    `_MISC_NPZ_NAME` above) plus the pure-constant computed tables (F3 SS5).
    """
    raw = _load_npz(_COLLISION_NPZ_NAME)
    raw.update(_load_npz(_MISC_NPZ_NAME))

    osm_nh42so4, osm_sodchl = init_osmo_par()

    return AmpsLuts(
        drpdrp=raw["drpdrp"],
        adrpdrp=_col_lut_aux(raw["drpdrp_aux"]),
        hexdrp=raw["hexdrp"],
        ahexdrp=_col_lut_aux(raw["hexdrp_aux"]),
        bbcdrp=raw["bbcdrp"],
        abbcdrp=_col_lut_aux(raw["bbcdrp_aux"]),
        coldrp=raw["coldrp"],
        acoldrp=_col_lut_aux(raw["coldrp_aux"]),
        gp1drp=raw["gp1drp"],
        agp1drp=_col_lut_aux(raw["gp1drp_aux"]),
        gp4drp=raw["gp4drp"],
        agp4drp=_col_lut_aux(raw["gp4drp_aux"]),
        gp8drp=raw["gp8drp"],
        agp8drp=_col_lut_aux(raw["gp8drp_aux"]),
        pol_frq=raw["pol_frq"],
        pla_frq=raw["pla_frq"],
        col_frq=raw["col_frq"],
        ros_frq=raw["ros_frq"],
        ppo_frq=raw["ppo_frq"],
        frq_aux=_grid_aux(raw["pol_frq_aux"]),
        mtac_map_col=raw["mtac_map_col"],
        mtac_map_pla=raw["mtac_map_pla"],
        map_col_aux=_grid_aux(raw["mtac_map_col_aux"]),
        map_pla_aux=_grid_aux(raw["mtac_map_pla_aux"]),
        lmt_mass_col=raw["lmt_mass_col"],
        lmt_mass_pla=raw["lmt_mass_pla"],
        lmt_mass_col_aux=_grid_aux(raw["lmt_mass_col_aux"]),
        lmt_mass_pla_aux=_grid_aux(raw["lmt_mass_pla_aux"]),
        znorm=raw["znorm"],
        osm_nh42so4=osm_nh42so4,
        osm_sodchl=osm_sodchl,
        snrml=init_normal_lut(),
        isnrml=init_inv_normal_lut(),
        vigp=init_inherent_growth_par(),
    )


# ---------------------------------------------------------------------------
# Osmotic-coefficient LUTs, init_osmo_par (mod_amps_utility.F90:12923-12986),
# osm_ammsul (mod_amps_utility.F90:12988-13051), osm_sodchl
# (mod_amps_utility.F90:13054-13085), F3 SS5.1.
#
# F3 SS5.1 only paraphrased the interpolation-node x-grid ("x = (/
# 0.0,0.1,0.2,...,5.5 /) (23 pts)", internally inconsistent -- see the
# superseded NEEDS_CONTEXT note this replaces, in the M1 Task 5 report).
# Per coordinator authorization, read `init_osmo_par`/`osm_ammsul`/
# `osm_sodchl` directly in mod_amps_utility.F90 (~line 12923) to settle it.
# The real x-grid is NOT uniform: dense (0.1) up to 1.0, medium (0.2) up to
# 2.0, coarse (0.5) up to the domain end. Quoted verbatim below (both `x`
# and `y` are literal Fortran array constructors in the real source, unlike
# the earlier paraphrase).
# ---------------------------------------------------------------------------

# osm_ammsul (mod_amps_utility.F90:12988-13051), verbatim:
#   x = (/ 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,
#          2.5,3.0,3.5,4.0,4.5,5.0,5.5/)
#   y = (/ 1.0,0.767,0.731,0.707,0.690,0.677,0.667,0.658,0.652,0.646,0.640,
#          0.632,0.628,0.624,0.623,0.623,0.626,0.635,0.647,0.660,0.673,
#          0.686,0.699/)
_OSM_AMMSUL_X = np.array(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
    ],
    dtype=np.float64,
)
_OSM_AMMSUL_Y = np.array(
    [
        1.0,
        0.767,
        0.731,
        0.707,
        0.690,
        0.677,
        0.667,
        0.658,
        0.652,
        0.646,
        0.640,
        0.632,
        0.628,
        0.624,
        0.623,
        0.623,
        0.626,
        0.635,
        0.647,
        0.660,
        0.673,
        0.686,
        0.699,
    ],
    dtype=np.float64,
)

# osm_sodchl (mod_amps_utility.F90:13054-13085), verbatim: same 23-point
# prefix as osm_ammsul's x-grid, plus one more node at 6.0.
#   x = (/ 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,
#          2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0/)
#   y = (/ 1.0,0.932,0.925,0.922,0.920,0.921,0.923,0.926,0.929,0.932,0.936,
#          0.943,0.951,0.962,0.972,0.983,1.013,1.045,1.080,1.116,1.153,
#          1.192,1.231,1.271/)
_OSM_SODCHL_X = np.concatenate([_OSM_AMMSUL_X, np.array([6.0], dtype=np.float64)])
_OSM_SODCHL_Y = np.array(
    [
        1.0,
        0.932,
        0.925,
        0.922,
        0.920,
        0.921,
        0.923,
        0.926,
        0.929,
        0.932,
        0.936,
        0.943,
        0.951,
        0.962,
        0.972,
        0.983,
        1.013,
        1.045,
        1.080,
        1.116,
        1.153,
        1.192,
        1.231,
        1.271,
    ],
    dtype=np.float64,
)


def _osm_ammsul(x: np.ndarray) -> np.ndarray:
    """(NH4)2SO4 osmotic coefficient, transcribed VERBATIM from
    `osm_ammsul` (`mod_amps_utility.F90:12988-13051`, F3 SS5.1): for
    `molality >= x(nt)`, `out = y(nt)` (flat, the LAST node value, not a
    linear extrapolation); for `molality < x(1)`, `out = y(1)`; otherwise
    linear interpolation between the bracketing nodes `x(i) <= molality <
    x(i+1)`. `np.interp`'s default behavior (flat-beyond-ends,
    linear-inside) matches this exactly given the verbatim `_OSM_AMMSUL_X`/
    `_OSM_AMMSUL_Y` node arrays above.
    """
    return np.interp(x, _OSM_AMMSUL_X, _OSM_AMMSUL_Y)


def _osm_sodchl(x: np.ndarray) -> np.ndarray:
    """NaCl osmotic coefficient, transcribed VERBATIM from `osm_sodchl`
    (`mod_amps_utility.F90:13054-13085`, F3 SS5.1); same
    flat-beyond-ends/linear-inside semantics as `_osm_ammsul`."""
    return np.interp(x, _OSM_SODCHL_X, _OSM_SODCHL_Y)


def init_osmo_par() -> tuple[Data1DLut, Data1DLut]:
    """Osmotic-coefficient LUTs, transcribed VERBATIM from `init_osmo_par`
    (`mod_amps_utility.F90:12923-12986`, F3 SS5.1): molality grids
    `0:5.5:0.1` ((NH4)2SO4) and `0:6.0:0.1` (NaCl), `n=(xmax-xmin)/dx+1`.
    Both the outer LUT-grid bookkeeping AND the underlying `osm_ammsul`/
    `osm_sodchl` interpolation nodes are now verbatim (read directly from
    mod_amps_utility.F90 per coordinator authorization; see those
    functions' docstrings) -- no remaining NEEDS_CONTEXT here.
    """
    xmin, xmax, dx = 0.0, 5.5, 0.1
    n = round((xmax - xmin) / dx) + 1
    x = xmin + dx * np.arange(n)
    osm_nh42so4 = Data1DLut(n=n, xs=xmin, dx=dx, y=_osm_ammsul(x))

    xmin, xmax, dx = 0.0, 6.0, 0.1
    n = round((xmax - xmin) / dx) + 1
    x = xmin + dx * np.arange(n)
    osm_sodchl = Data1DLut(n=n, xs=xmin, dx=dx, y=_osm_sodchl(x))

    return osm_nh42so4, osm_sodchl


# ---------------------------------------------------------------------------
# Normal / inverse-normal CDF LUTs (F3 SS5.2, SS5.3). F3 SS6 explicitly
# sanctions scipy equivalents here: "osmotic and normal/inverse-normal LUTs
# and IGP splines are pure constants + scipy equivalents (scipy.stats.
# norm.sf, norm.ppf)".
# ---------------------------------------------------------------------------


def init_normal_lut() -> Data1DLut:
    """Standard normal survival-function LUT, `init_normal_lut`
    (`mod_amps_utility.F90:13087-13129`, F3 SS5.2): `x = 0:5:0.01` (n=501),
    `y_snrml(i) = 1 - Phi(x)` via `cdfnor` (DCDFLIB) -- `scipy.stats.
    norm.sf` is the F3-sanctioned equivalent (F3 SS6)."""
    n, xs, dx = 501, 0.0, 0.01
    x = xs + dx * np.arange(n)
    y = scipy.stats.norm.sf(x)
    return Data1DLut(n=n, xs=xs, dx=dx, y=y)


def init_inv_normal_lut() -> Data1DLut:
    """Inverse standard normal CDF LUT, `init_inv_normal_lut`
    (`mod_amps_utility.F90:13131-13187`, F3 SS5.3): log10-spaced
    probability grid `y` in `[1e-30, 0.5]`, `n=501`,
    `dy=(log10(ymax)-log10(ymin))/(n-1)`; `x(i)=dinvnr(y, 1-y)`.
    `scipy.stats.norm.ppf` is the F3-sanctioned equivalent of DCDFLIB's
    `dinvnr` here (F3 SS6). Stored per `data1d_lut_big`:
    `xs_isnrml=log10(ymin)=-30`, `dx_isnrml=dy`, `y_isnrml=x(1:n)`
    (`y` here is genuinely the LUT's stored VALUE array -- the log10-
    probability is the implicit coordinate, per the Fortran naming)."""
    n = 501
    ymin, ymax = 1.0e-30, 0.5
    xs = math.log10(ymin)
    dx = (math.log10(ymax) - math.log10(ymin)) / (n - 1)
    log10_prob = xs + dx * np.arange(n)
    prob = 10.0**log10_prob
    y = scipy.stats.norm.ppf(prob)
    return Data1DLut(n=n, xs=xs, dx=dx, y=y)


# ---------------------------------------------------------------------------
# Inherent Growth Parameterization splines, init_inherent_growth_par
# (mod_amps_utility.F90:1247-1321, F3 SS5.4) -- transcribed VERBATIM, all
# constants hard-coded, no NEEDS_CONTEXT.
# ---------------------------------------------------------------------------

NOK_IGP = 23

_X_IGP = np.array(
    [
        -60.0,
        -55.0,
        -50.0,
        -45.0,
        -40.0,
        -35.0,
        -30.0,
        -27.0,
        -25.0,
        -23.0,
        -21.0,
        -20.0,
        -17.0,
        -15.0,
        -12.0,
        -10.0,
        -8.0,
        -6.0,
        -5.0,
        -4.0,
        -3.5,
        -2.5,
        -1.5,
    ]
)  # F3 SS5.4, x_igp, 23 knots (deg C)

_A_IGP = np.array(
    [
        [0.000000e00, 0.000000e00, -3.000000e-02, 2.300000e00],
        [0.000000e00, 0.000000e00, -3.000000e-02, 2.150000e00],
        [-4.800000e-04, 2.400000e-03, -3.000000e-02, 2.000000e00],
        [1.586667e-03, -1.353333e-02, -4.200000e-02, 1.850000e00],
        [1.512821e-03, -5.897436e-03, -5.833333e-02, 1.500000e00],
        [-9.597381e-05, 8.490998e-04, -3.846154e-03, 1.250000e00],
        [-1.176598e-04, 9.293226e-05, -2.553191e-03, 1.240000e00],
        [2.040230e-03, -6.494253e-03, -5.172414e-03, 1.230000e00],
        [-1.439394e-03, 3.712121e-03, -6.666667e-03, 1.210000e00],
        [-1.597052e-03, -1.726044e-02, -9.090909e-03, 1.200000e00],
        [4.920791e-01, -7.947818e-01, -9.729730e-02, 1.100000e00],
        [-0.0012, 0.0363, -0.2244, 0.7000],
        [-0.0005, 0.0141, -0.0511, 0.3200],
        [0.0035, 0.0109, -0.0011, 0.2700],
        [-0.0063, 0.0427, 0.1597, 0.4600],
        [0.0212, 0.0051, 0.2552, 0.9000],
        [-0.1109, 0.1320, 0.5294, 1.6000],
        [0.1061, -0.5332, -0.2729, 2.3000],
        [0.2758, -0.2148, -1.0209, 1.6000],
        [-0.0118, 0.6125, -0.6233, 0.6400],
        [-0.2951, 0.5948, -0.0197, 0.4800],
        [0.0550, -0.0995, 0.1244, 0.7600],
        [-0.0000, 0.0108, 0.0906, 0.8400],
    ]
)  # F3 SS5.4, a_igp(1:23, 1:4)

_B_IGP_COLD = np.array(
    [
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0, 0.0, 0.0, 0.7600],
        [0.0431, -0.1031, 0.0, 0.7600],
    ]
)  # F3 SS5.4, b_igp(1:11, 1:4)


def init_inherent_growth_par() -> VapIgpAux:
    """Inherent Growth Parameterization spline knots, transcribed VERBATIM
    from `init_inherent_growth_par` (`mod_amps_utility.F90:1247-1321`,
    F3 SS5.4). `b_igp(12:23,1:4) = a_igp(12:23,1:4)` per the Fortran
    (line 639)."""
    b = np.concatenate([_B_IGP_COLD, _A_IGP[11:]], axis=0)
    return VapIgpAux(nok=NOK_IGP, x=_X_IGP.copy(), a=_A_IGP.copy(), b=b)


# ---------------------------------------------------------------------------
# Low-List collisional-breakup fragment tables, cal_breakfragment
# (mod_amps_lib.F90:1831-2017, F3 SS5.5) -- SIZE/allocation only, see
# module docstring NEEDS_CONTEXT.
# ---------------------------------------------------------------------------


def breakup_fragment_table_sizes(nrbin: int, jmin_bk: int) -> tuple[int, int]:
    """Low-List collisional-breakup fragment-table SIZES, transcribed
    VERBATIM from `cal_breakfragment` (`mod_amps_lib.F90:1831-2017`,
    F3 SS5.5):

        imin_bk = jmin_bk + 1
        imax_bk = NRBIN
        i1d_pair_max = (imax_bk-1) - jmin_bk + 1 \
            + (imax_bk-imin_bk)*(1+imax_bk-imin_bk)/2
        kk_max = i1d_pair_max * NRBIN   ! liquid%N_BIN == NRBIN

    `jmin_bk` ("bin that has the minimum size for possible breakup": the
    smallest liquid-bin index `i` with `liquid%MS(i,1)%len >= D_0=0.01cm`)
    is a REQUIRED argument, NOT derived here -- F3 does not give the mass
    -> diameter formula needed to compute it (NEEDS_CONTEXT; same
    "no compile-time default, caller must supply it" pattern `bin_grid.py`
    uses for `nbin_h`).

    Cross-checked against F3 SS4's quoted `class_Cloud_Micro.F90`
    declaration `bu_fd(2,62400), bu_tmass(780)` ("this is for 80 bins"):
    `breakup_fragment_table_sizes(80, 41) == (780, 62400)` reproduces it
    exactly -- the one concrete `(NRBIN, sizes)` data point F3 provides.
    """
    imin_bk = jmin_bk + 1
    imax_bk = nrbin
    i1d_pair_max = (imax_bk - 1) - jmin_bk + 1 + (imax_bk - imin_bk) * (1 + imax_bk - imin_bk) // 2
    kk_max = i1d_pair_max * nrbin
    return i1d_pair_max, kk_max


def make_breakup_fragment_tables(nrbin: int, jmin_bk: int) -> tuple[np.ndarray, np.ndarray]:
    """Allocate the Low-List breakup fragment tables `bu_fd`, `bu_tmass`
    (F3 SS5.5), zero-initialized exactly as `cal_breakfragment` does before
    its fill loop (`bu_fd=0.0_PS; bu_tmass=0.0_PS`).

    NEEDS_CONTEXT: only the correctly-SIZED, zero-filled allocation is
    provided. The actual FILL (the Low-List fragment mass/count physics)
    calls `cal_Coalescence_Efficiency` and `cal_breakup_dis_LL`
    (`mod_amps_core.F90`), neither of which is quoted in F3 -- out of
    scope for this LUT-conversion task (a later physics-port task owns
    them). This satisfies this task's own test requirement ("breakup
    tables sized per bin count formula from F3"); it does NOT compute
    physically meaningful fragment distributions.

    Returns:
        (bu_fd, bu_tmass): `bu_fd` has shape `(2, kk_max)`, `bu_tmass` has
        shape `(i1d_pair_max,)`, per `breakup_fragment_table_sizes`.
    """
    i1d_pair_max, kk_max = breakup_fragment_table_sizes(nrbin, jmin_bk)
    bu_fd = np.zeros((2, kk_max), dtype=np.float64)
    bu_tmass = np.zeros(i1d_pair_max, dtype=np.float64)
    return bu_fd, bu_tmass

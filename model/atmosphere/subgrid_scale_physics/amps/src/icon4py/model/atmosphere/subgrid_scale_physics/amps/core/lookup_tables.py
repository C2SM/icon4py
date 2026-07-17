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
  from the real AMPS_DATA files into the packaged `data/amps_luts.npz`
  (see `data/README.md` for provenance), loaded here via
  `importlib.resources` (F5 SS3 packaged-data pattern).
* Computed at load time, no data file (F3 SS5): osmotic-coefficient LUTs
  (`init_osmo_par`), the normal/inverse-normal CDF LUTs
  (`init_normal_lut`/`init_inv_normal_lut`), and the Inherent Growth
  Parameterization spline knots (`init_inherent_growth_par`) -- all pure
  constants, transcribed VERBATIM where F3 quotes exact Fortran source.

One exception, NOT computed as part of `load_luts()`: the Low-List
collisional-breakup fragment tables (`bu_fd`/`bu_tmass`, F3 SS5.5). F3's
quoted `cal_breakfragment` needs a runtime bin-count `NRBIN` (config-
dependent, unknown here) AND `jmin_bk` (the smallest liquid bin whose
diameter clears the `D_0=0.01cm` breakup cutoff) -- the latter has no
F3-quoted formula at all (it depends on the liquid Group's mass -> diameter
conversion, not given anywhere in F3; NEEDS_CONTEXT). `jmin_bk` is
therefore a REQUIRED argument to `breakup_fragment_table_sizes`/
`make_breakup_fragment_tables` below, the same "no compile-time default,
caller must supply it" pattern `bin_grid.py` already uses for `nbin_h`
(see that module's docstring). Only the SIZE/allocation of `bu_fd`/
`bu_tmass` is provided (zero-filled, matching the Fortran pre-loop state
`bu_fd=0.0_PS; bu_tmass=0.0_PS`); the actual fragment-mass/-count physics
fill calls `cal_Coalescence_Efficiency` + `cal_breakup_dis_LL`
(`mod_amps_core.F90`), neither quoted in F3 -- out of scope for this LUT-
conversion task.

A second, narrower NEEDS_CONTEXT applies to `init_osmo_par`'s two
interpolation-curve helper functions (`_osm_ammsul`/`_osm_sodchl`): F3
quotes their y-data verbatim but only PARAPHRASES the interpolation-node
x-grid (internally inconsistent -- see `_osm_ammsul`'s docstring). The
x-grid used here is a documented, unique-under-the-given-constraints
INFERENCE, not a verbatim transcription.
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
_NPZ_NAME = "amps_luts.npz"


def _col_lut_aux(raw: np.ndarray) -> ColLutAux:
    xs, dx, ys, dy, nr, nc = raw
    return ColLutAux(xs=float(xs), dx=float(dx), ys=float(ys), dy=float(dy), nr=int(nr), nc=int(nc))


def _grid_aux(raw: np.ndarray) -> GridAux:
    nr, nc = raw
    return GridAux(nr=int(nr), nc=int(nc))


def load_luts() -> AmpsLuts:
    """Load the packaged AMPS_DATA-derived lookup tables (via
    `importlib.resources` on this package's `data/` directory) plus the
    pure-constant computed tables (F3 SS5)."""
    resource = importlib.resources.files(_PACKAGE).joinpath(_NPZ_NAME)
    with importlib.resources.as_file(resource) as path, np.load(path) as npz:
        raw = {key: npz[key] for key in npz.files}

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
# Osmotic-coefficient LUTs, init_osmo_par (mod_amps_utility.F90:12923-12986,
# F3 SS5.1).
# ---------------------------------------------------------------------------

# F3 SS5.1: "Hard-coded curve data (osm_ammsul): x = (/ 0.0,0.1,0.2,...,5.5 /)
# (23 pts)". The y-data below is a verbatim transcription (23 comma-
# separated literals, F3's own quote). The x-line is NOT verbatim: it is an
# ellipsis paraphrase, not valid Fortran array-literal syntax (unlike the y
# line right next to it), and it is internally inconsistent -- 0.0 to 5.5 in
# steps of 0.1 would be 56 points, not 23.
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
)  # F3 SS5.1, osm_ammsul y-data, 23 points

# F3 SS5.1: "osm_sodchl: 24 pts" -- same verbatim-y/paraphrased-x situation.
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
)  # F3 SS5.1, osm_sodchl y-data, 24 points


def _osm_ammsul(x: np.ndarray) -> np.ndarray:
    """(NH4)2SO4 osmotic coefficient: piecewise-linear interpolation of the
    experimental curve, flat extrapolation beyond the ends (F3 SS5.1: "flat
    extrapolation beyond ends, linear interpolation inside").

    NEEDS_CONTEXT: the true interpolation-node x-grid is not verbatim in
    F3 (see module docstring). The x-grid used here,
    `np.linspace(0.0, 5.5, 23)`, is the UNIQUE uniformly-spaced grid
    consistent with the only hard facts F3 gives (23 points, curve's own
    stated domain 0.0 to 5.5 -- the same `xmax` `init_osmo_par` uses for
    the (NH4)2SO4 LUT) -- a documented inference, not a transcription.
    `np.interp` already implements flat-extrapolation-beyond-ends /
    linear-inside, matching F3's stated behavior exactly.
    """
    x_nodes = np.linspace(0.0, 5.5, len(_OSM_AMMSUL_Y), dtype=np.float64)
    return np.interp(x, x_nodes, _OSM_AMMSUL_Y)


def _osm_sodchl(x: np.ndarray) -> np.ndarray:
    """NaCl osmotic coefficient; same NEEDS_CONTEXT caveat as
    `_osm_ammsul` (F3 SS5.1) -- x-grid inferred as
    `np.linspace(0.0, 6.0, 24)` (24 points, domain 0.0 to 6.0, the `xmax`
    `init_osmo_par` uses for the NaCl LUT)."""
    x_nodes = np.linspace(0.0, 6.0, len(_OSM_SODCHL_Y), dtype=np.float64)
    return np.interp(x, x_nodes, _OSM_SODCHL_Y)


def init_osmo_par() -> tuple[Data1DLut, Data1DLut]:
    """Osmotic-coefficient LUTs, transcribed from `init_osmo_par`
    (`mod_amps_utility.F90:12923-12986`, F3 SS5.1): molality grids
    `0:5.5:0.1` ((NH4)2SO4) and `0:6.0:0.1` (NaCl), `n=(xmax-xmin)/dx+1`.
    The grid bookkeeping is verbatim; the underlying `osm_ammsul`/
    `osm_sodchl` interpolation nodes are a documented inference -- see
    their docstrings (NEEDS_CONTEXT).
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

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The dispersion relation of the linear and quadratic schemes (Jocksch et al., PPAM 2026, section 4).

Figures 5 and 6 of the paper (phase speed and growth rate of a plane wave against its
wavenumber for CFL 0.01, 0.1, ..., 0.5) and the theta = 0 half of figure 7 (growth rate
over the (CFL, wavenumber) plane), for the schemes 2 (Miura, linear reconstruction) and 3
(Miura, quadratic), unlimited: the limiters are nonlinear and the real and imaginary parts
of the wave are two real tracers whose responses must superpose.

The setup is A. Jocksch's live block in mo_nh_stepping.f90 of icon-exclaim branch
transport_ajocksch (the 'if (.false.)' block at lines 3344-3488; the numbers are of the
worktree icon-ajocksch of the tracer-advection port workspace), which produced his tables
dispersion_linear.txt (scheme 2) and dispersion_quadratic.txt (scheme 3):

- his grid torus_grid_r4_c200_elen100.nc (880 cells, 5 km edges, centred coordinates), a
  uniform wind |v| = 1 along x (angle 0 at :3345-3347), the edge mass flux ``vn_traj =
  mass_flx_me = u n_x`` (his edge-vector formula :3349-3405 equals the edge normal on his
  grid, every ``n_x >= 0``), air mass 1 and no vertical flux (:3406-3411);
- CFL = icfl / 100 for icfl = 0, 2, ..., 100 with 0 -> 0.01 (:3423-3425); wavenumber
  alpha = ireal / 100 pi for ireal = 0, 2, ..., 100 (:3427); the wave
  ``exp(-i x 2 / a alpha)`` on the cell centres as tracers 1 (real) and 2 (imaginary),
  :3429-3436, set fresh for every (CFL, alpha);
- one step_advection with ``dt = (a / 2) 1.9999999999 CFL`` (:3440), so CFL is the Courant
  number with respect to the edge length ``a`` (the wind covers ``2 CFL`` half-edges);
- ``omega = -ln(q_new / q_now) i / dt (a / 2)`` at the Fortran cell 883/3 = 294 (:3462,
  1-based; cell index 293 here) and ``diff = |q_new - q_now / exp(-i 2 CFL alpha)|``
  (:3469-3478); ``print '(4F20.16)'`` of CFL, alpha, Re omega, Im omega (:3479; the
  fifth item, diff, wraps onto a second line under this format in his ``_new`` tables,
  and is absent from dispersion_linear/quadratic.txt); a blank line per CFL block.
  The older variant of the block (dispersion_relation/MO_NH_STEPPING/
  mo_nh_stepping.f90_dispersion:2158-2183) samples cell 880/3 = 293, normalises by cell
  291 and prints ``Re omega + 2 pi``; his tables carry the raw principal value (CFL 1,
  alpha = pi: -1e-16), so it is the live block that is mirrored here.

Here one alpha per vertical level (his own device in the theta = 30 degree block :3489-;
the columns of the granule are independent, which the test asserts by re-running one
CFL with a single alpha on every level), so the 51 x 51 table costs 51 CFL x 2 tracers
``Advection.run`` calls per scheme. The granule is the one the driver builds on his grid
file, with the states of the cylinder experiment (jocksch_cylinder.yaml) and the
metrics of a flat 51-level column; ``grf_tend_tracer`` is zero here, -1 in the Fortran
(:3412, unused on a torus). Outputs (git-ignored, under weno_data/dispersion of the
workspace): his four-column table per scheme, the per-cell frequencies, a summary with
the tip-up / tip-down spread and the growth boundary, and the figures.

Gates: the frequency at his cell against his table, per (CFL, alpha), see
FORTRAN_TOLERANCE (measured on gtfn_cpu, values in the comments), the level independence
bit-exact, and the exact-translation check ``diff`` of the longest wave (alpha = pi/50)
small against the value a wave carried the wrong way or not at all would give
(``2 sin(CFL alpha)``; the sign of the phase and of the wind).

Measured (gtfn_cpu, the full table; dace_cpu for CFL 0.2 gives the same digits):
- his cell 294 is a tip-up cell; the cells of one parity at least three edge lengths
  from the periodic seam in x agree among themselves to 5e-13 / 2e-13 (scheme 2, Re /
  Im, worst at CFL 0.38) and 1.4e-11 / 8e-11 (scheme 3, CFL 0.48), Re taken modulo the
  aliasing period ``2 pi (a / 2) / dt`` (the principal branch flips between cells by
  round-off where ``2 CFL alpha`` is near pi), and the two parities' means agree to
  9e-16 / 4e-16 (scheme 2) and 5e-14 / 2.4e-13 (scheme 3): at theta = 0 the sampling
  cell's parity does not matter, cell 293 or 294 is the same curve. The wave is
  periodic on the 100 km torus only for ireal a multiple of 10 (alpha = n pi / 20), so
  for the other alphas the seam between x = 47.5 km and -52.5 km is a discontinuity
  that contaminates the cells within reach of the stencil: 66 cells (three columns of
  22) for scheme 2, 110 (five columns) for scheme 3, for every CFL; his cell
  (x = -17.5 km) is 7 cells away. The summary files carry the interior spread and the
  count of deviating cells per CFL.
- growth (-Im omega > 0 at his cell): none for CFL <= 0.68 (scheme 2) / 0.60 (scheme 3);
  scheme 2 grows for CFL >= 0.70 from alpha = 0 up to 1.27 (CFL 0.70), 2.15 (0.72),
  2.69 (0.74) and for every alpha from CFL 0.76 on (max rate 0.35 at CFL 1); scheme 3
  grows for CFL >= 0.62 in the band alpha in (0, 0.38) widening to (0, 1.80) at CFL 1,
  at rates below 4.4e-3. Figures 5 and 6 are reproduced for the six CFLs of the paper.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import time as wall_time
from typing import Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection, tracer_advection_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry_attributes as geometry_attrs
from icon4py.model.common.initial_condition.analytical import (
    linear_horizontal_advection as lin_hor_adv_ic,
    plane_wave,
)
from icon4py.model.common.states import adv_states
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import config as test_config
from icon4py.model.testing.fixtures.datatest import backend, process_props

from .test_jocksch_reference import LIMITERS, SCHEMES


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # the figures are optional, the tables are not
    plt = None


WORKSPACE: Final = pathlib.Path("/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim")
#: his torus (a copy of dispersion_relation/icon/grids/torus_grid_r4_c200_elen100.nc)
GRID_FILE: Final = WORKSPACE / "weno_data" / "grids" / "jocksch_torus_grid_r4_c200_elen100.nc"
OUTPUT_DIR: Final = WORKSPACE / "weno_data" / "dispersion"
#: his tables, F20.16 x 4 (CFL, alpha, Re omega, Im omega), a blank line per CFL block
REFERENCE_TABLES: Final[dict[int, pathlib.Path]] = {
    2: pathlib.Path(
        "/capstor/scratch/cscs/ajocksch/dispersion_relation/icon/run/dispersion_linear.txt"
    ),
    3: pathlib.Path(
        "/capstor/scratch/cscs/ajocksch/dispersion_relation/icon/run/dispersion_quadratic.txt"
    ),
}
EXPERIMENT_CONFIG: Final = test_config.EXPERIMENT_CONFIG_PATH / "jocksch_cylinder.yaml"

#: the edge length a of his grid; 'length' in the Fortran (the reference edge 695, :3350-3359)
EDGE_LENGTH: Final = 5000.0
#: dt = 1.0 * (length / 2) * 1.9999999999 * cfl, mo_nh_stepping.f90:3440
DT_FACTOR: Final = 1.9999999999
#: cfl = icfl / 100, icfl = 0, 2, ..., 100, 0 -> 0.01 (:3423-3425)
CFL_FULL: Final[tuple[float, ...]] = tuple(
    0.01 if icfl == 0 else icfl / 100 for icfl in range(0, 101, 2)
)
#: the CFL numbers of figures 5 and 6
CFL_PAPER: Final[tuple[float, ...]] = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5)
#: one row for a second backend
CFL_SINGLE: Final[tuple[float, ...]] = (0.2,)
CFL_SETS: Final[dict[str, tuple[float, ...]]] = {
    "paper": CFL_PAPER,
    "full": CFL_FULL,
    "single": CFL_SINGLE,
}
#: alpha = ireal / 100 * acos(-1), ireal = 0, 2, ..., 100 (:3427), one per level
ALPHA: Final[tuple[float, ...]] = tuple(ireal / 100 * math.pi for ireal in range(0, 101, 2))
NUM_LEVELS: Final = len(ALPHA)
#: his sampling cell 883/3 = 294 (1-based, :3462); it is a tip-up cell at (-17.5, 14.43) km
SAMPLE_CELL: Final = 883 // 3 - 1
#: the CFL and the level used for the level-independence check
INDEPENDENCE_CFL: Final = 0.2
INDEPENDENCE_LEVEL: Final = 25

#: max |Re omega - his| and max |Im omega - his| at his cell over the whole table, gated
#: at about three times the value measured on gtfn_cpu (in the comments; the scheme-3
#: values are the SVD round-off of its pseudoinverse, 7e-13 at L1, amplified by the
#: logarithm's 1 / (2 CFL |q_new / q_now|), worst at CFL 0.48)
FORTRAN_TOLERANCE: Final[dict[int, tuple[float, float]]] = {
    2: (5e-14, 1.5e-13),  # 1.5e-14, 4.9e-14 (CFL 0.38)
    3: (4e-11, 2e-10),  # 1.2e-11, 6.3e-11 (CFL 0.48)
}
#: the exact-translation error of the longest wave (alpha = pi/50) relative to the value
#: 2 sin(CFL alpha) a wave carried the wrong way or not at all would give; measured
#: 3e-5 (CFL 0.01) to 7e-4 (CFL 1)
TRANSLATION_TOLERANCE: Final = 0.1
#: cells at least this many edge lengths from the periodic seam in x count as interior
SEAM_MARGIN: Final = 3

_SCHEME_IDS: Final = {2: "ihadv2", 3: "ihadv3"}


@dataclasses.dataclass(frozen=True)
class _Setup:
    advection: tracer_advection.Advection
    grid: object
    backend: gtx_typing.Backend | None
    cell_x: np.ndarray
    cell_y: np.ndarray
    tip_up: np.ndarray
    prep_adv: adv_states.AdvectionPrepAdvState
    diagnostic_state: tracer_advection_states.AdvectionDiagnosticState


_setups: dict[tuple[int, str], _Setup] = {}


def _classify_tip_up(cell_y: np.ndarray) -> np.ndarray:
    """True for the tip-up cells (base below the centre), from the rows of cell centres.

    In every strip of triangles the tip-up centres sit a third of the height above the
    base and the tip-down ones two thirds, so the centre rows alternate with the gaps
    h/3, 2h/3, h/3, ...; anchored at the sampling cell, which is tip-up (its vertices in
    the grid file: two at its minimum y, one at the maximum).
    """
    height = EDGE_LENGTH * math.sqrt(3.0) / 2.0
    steps = np.rint((cell_y - cell_y[SAMPLE_CELL]) / (height / 3.0)).astype(int) % 3
    assert set(np.unique(steps)) <= {0, 1}, "cell-centre rows are not on the expected lattice"
    return steps == 0


def _setup(
    scheme: int,
    backend: gtx_typing.Backend | None,
    process_props: decomposition.ProcessProperties,
    tmp_path: pathlib.Path,
) -> _Setup:
    """The granule the driver builds on his grid, the wind and the air mass of his block."""
    key = (scheme, data_alloc.backend_name(backend))
    if key in _setups:
        return _setups[key]
    config = config_io.read_yaml_str(
        EXPERIMENT_CONFIG.read_text(), driver_config.ExperimentConfig
    ).with_overrides(
        vertical_grid={"num_levels": NUM_LEVELS},
        driver={"output_path": tmp_path / "driver_output", "enable_output": False},
        tracer_advection={
            "horizontal_advection_type": SCHEMES[scheme],
            "horizontal_advection_limiter": LIMITERS[0],
        },
    )
    allocator = model_backends.get_allocator(backend)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=GRID_FILE,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    icon4py_driver = driver.initialize_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    assert icon4py_driver.granules.tracer_advection is not None
    grid = grid_manager.grid
    geometry = icon4py_driver.static_field_factories.geometry
    cell_x = geometry.get(geometry_attrs.CELL_CENTER_X).asnumpy()
    cell_y = geometry.get(geometry_attrs.CELL_CENTER_Y).asnumpy()
    edge_length = geometry.get(geometry_attrs.EDGE_LENGTH).asnumpy()
    np.testing.assert_allclose(edge_length, EDGE_LENGTH, rtol=1e-12)
    normal_x = geometry.get(geometry_attrs.EDGE_NORMAL_U).ndarray
    normal_y = geometry.get(geometry_attrs.EDGE_NORMAL_V).ndarray
    # his edge-vector mass flux is u n_x on this grid, with n_x >= 0 on every edge
    assert (data_alloc.as_numpy(normal_x) >= 0.0).all()

    prep_adv = adv_states.AdvectionPrepAdvState(
        vn_traj=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend),
        mass_flx_me=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend),
        mass_flx_ic=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        ),
    )
    # angle 0: u_x = 1, u_y = 0 (:3345-3347); vn_traj = mass_flx_me, mass_flx_ic = 0 (:3406-3408)
    lin_hor_adv_ic.prescribe_uniform_wind(
        prep_adv_state=prep_adv,
        primal_normal_x=normal_x,
        primal_normal_y=normal_y,
        u=1.0,
        v=0.0,
    )
    # airmass_now = airmass_new = 1 (:3410-3411); grf_tend_tracer is unused on a torus
    diagnostic_state = tracer_advection_states.AdvectionDiagnosticState(
        airmass_now=data_alloc.constant_field(
            grid, 1.0, dims.CellDim, dims.KDim, allocator=backend
        ),
        airmass_new=data_alloc.constant_field(
            grid, 1.0, dims.CellDim, dims.KDim, allocator=backend
        ),
        grf_tend_tracer=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
        hfl_tracer=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend),
        vfl_tracer=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        ),
    )
    _setups[key] = _Setup(
        advection=icon4py_driver.granules.tracer_advection,
        grid=grid,
        backend=backend,
        cell_x=cell_x,
        cell_y=cell_y,
        tip_up=_classify_tip_up(cell_y),
        prep_adv=prep_adv,
        diagnostic_state=diagnostic_state,
    )
    return _setups[key]


def _advect(setup: _Setup, q_now: np.ndarray, dtime: float) -> np.ndarray:
    """One Advection.run of the (cells, levels) tracer, returned as a numpy array."""
    p_tracer_now = gtx.as_field((dims.CellDim, dims.KDim), q_now, allocator=setup.backend)
    p_tracer_new = data_alloc.zero_field(
        setup.grid, dims.CellDim, dims.KDim, allocator=setup.backend
    )
    setup.advection.run(
        diagnostic_state=setup.diagnostic_state,
        prep_adv=setup.prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=dtime,
    )
    return p_tracer_new.asnumpy().copy()


def _time_step(cfl: float) -> float:
    return 1.0 * (EDGE_LENGTH / 2) * DT_FACTOR * cfl


def _initial_wave(setup: _Setup, alphas: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary tracers (cells, levels), level k carrying the wave of alphas[k]."""
    re = np.empty((setup.cell_x.size, len(alphas)))
    im = np.empty_like(re)
    for k, alpha in enumerate(alphas):
        re[:, k], im[:, k] = plane_wave.plane_wave(
            alpha=alpha,
            wind_angle=0.0,
            cell_center_x=setup.cell_x,
            cell_center_y=setup.cell_y,
            edge_length=EDGE_LENGTH,
        )
    return re, im


@dataclasses.dataclass(frozen=True)
class _Table:
    cfls: tuple[float, ...]
    #: (cfl, cell, alpha) complex frequencies and exact-translation errors
    omega: np.ndarray
    diff: np.ndarray
    #: (cfl, cell, alpha) the advected complex wave (for the backend comparison)
    q_new: np.ndarray
    seconds_per_run: float


def _dispersion(setup: _Setup, cfls: tuple[float, ...]) -> _Table:
    re0, im0 = _initial_wave(setup, ALPHA)
    q_now = re0 + 1j * im0
    alpha = np.asarray(ALPHA)[None, :]
    omega = np.empty((len(cfls), q_now.shape[0], NUM_LEVELS), dtype=complex)
    diff = np.empty(omega.shape)
    q_new_all = np.empty(omega.shape, dtype=complex)
    start = wall_time.perf_counter()
    for i, cfl in enumerate(cfls):
        dtime = _time_step(cfl)
        q_new = _advect(setup, re0, dtime) + 1j * _advect(setup, im0, dtime)
        omega[i] = plane_wave.numerical_frequency(
            q_now=q_now, q_new=q_new, dtime=dtime, edge_length=EDGE_LENGTH
        )
        diff[i] = plane_wave.exact_translation_error(q_now=q_now, q_new=q_new, cfl=cfl, alpha=alpha)
        q_new_all[i] = q_new
    seconds = (wall_time.perf_counter() - start) / (2 * len(cfls))
    return _Table(cfls=cfls, omega=omega, diff=diff, q_new=q_new_all, seconds_per_run=seconds)


def _read_fortran_table(path: pathlib.Path) -> np.ndarray:
    """(rows, 4) of his table, the blank block separators dropped."""
    rows = [
        [float(item) for item in line.split()]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return np.asarray(rows)


def _fortran_block(table: np.ndarray, cfl: float) -> np.ndarray:
    """(51, 4) rows of one CFL of his table, in his alpha order (asserted to be ALPHA)."""
    block = table[np.isclose(table[:, 0], cfl, rtol=0.0, atol=1e-15)]
    assert block.shape[0] == NUM_LEVELS, f"CFL {cfl}: {block.shape[0]} rows in the reference"
    np.testing.assert_allclose(block[:, 1], ALPHA, rtol=0.0, atol=1e-15)
    return block


def _write_fortran_format(path: pathlib.Path, table: _Table, cell: int) -> None:
    """His table: print '(4F20.16)' cfl, alpha, Re omega, Im omega; a blank line per block."""
    lines = []
    for i, cfl in enumerate(table.cfls):
        for k, alpha in enumerate(ALPHA):
            omega = table.omega[i, cell, k]
            lines.append(f"{cfl:20.16f}{alpha:20.16f}{omega.real:20.16f}{omega.imag:20.16f}")
        lines.append(" ")
    path.write_text("\n".join(lines) + "\n")


def _growth_boundary(alphas: np.ndarray, growth: np.ndarray) -> list[float]:
    """The alphas where the growth rate -Im omega crosses zero (linear interpolation)."""
    crossings = []
    for k in range(len(alphas) - 1):
        g0, g1 = growth[k], growth[k + 1]
        if (g0 > 0.0) != (g1 > 0.0) and g0 != g1:
            crossings.append(float(alphas[k] + (alphas[k + 1] - alphas[k]) * g0 / (g0 - g1)))
    return crossings


def _interior(cell_x: np.ndarray) -> np.ndarray:
    """The cells at least SEAM_MARGIN edge lengths from the periodic seam in x."""
    margin = SEAM_MARGIN * EDGE_LENGTH
    return (cell_x >= cell_x.min() + margin) & (cell_x <= cell_x.max() - margin)


def _deviation(omega: np.ndarray, reference: np.ndarray, cfl: float) -> np.ndarray:
    """omega - reference (cells, alpha), Re taken modulo the aliasing period 2 pi (a/2) / dt."""
    period = 2.0 * math.pi * (EDGE_LENGTH / 2) / _time_step(cfl)
    d_re = omega.real - reference.real
    d_re -= period * np.rint(d_re / period)
    return d_re + 1j * (omega.imag - reference.imag)


def _summarise(
    table: _Table, setup: _Setup, reference: np.ndarray | None
) -> tuple[list[str], float, float]:
    """The per-CFL summary lines; returns (lines, max |dRe|, max |dIm|) against his table."""
    alphas = np.asarray(ALPHA)
    up = setup.tip_up
    interior = _interior(setup.cell_x)
    lines = [
        "# per CFL: max over alpha of |Re omega - Fortran|, |Im omega - Fortran| at his cell "
        f"{SAMPLE_CELL} (1-based {SAMPLE_CELL + 1}, tip-up); the spread (max - min over "
        f"cells, Re modulo the aliasing period) of Re/Im omega among the {int((interior & up).sum())} "
        f"tip-up and the {int((interior & ~up).sum())} tip-down cells at least {SEAM_MARGIN} "
        "edge lengths from the periodic seam in x, and between the two parities' interior "
        "means, max over alpha; the number of the 880 cells deviating by more than 1e-9 from "
        "their parity's mean (the seam's reach); max diff (exact translation) at his cell; "
        "the growth rate -Im omega at his cell: max over alpha, the alpha of the max, the "
        "zero crossings",
        "# cfl  dRe  dIm  spread_up_re  spread_up_im  spread_down_re  spread_down_im  "
        "up_vs_down_re  up_vs_down_im  n_seam  diff_max  growth_max  alpha_at_growth_max  "
        "growth_zero_alphas",
    ]
    max_d_re = 0.0
    max_d_im = 0.0
    for i, cfl in enumerate(table.cfls):
        omega = table.omega[i]  # (cells, alpha)
        at_cell = omega[SAMPLE_CELL]
        if reference is not None:
            block = _fortran_block(reference, cfl)
            d_re = float(np.abs(at_cell.real - block[:, 2]).max())
            d_im = float(np.abs(at_cell.imag - block[:, 3]).max())
        else:
            d_re = d_im = math.nan
        max_d_re = max(max_d_re, d_re)
        max_d_im = max(max_d_im, d_im)

        parity_mean = {}
        spread = {}
        n_seam = 0
        for name, parity in (("up", up), ("down", ~up)):
            anchor = omega[np.flatnonzero(interior & parity)[0]]
            deviation = _deviation(omega[parity], anchor, cfl)
            parity_mean[name] = anchor + deviation[interior[parity]].mean(axis=0)
            centred = _deviation(omega[parity], parity_mean[name], cfl)
            inner = centred[interior[parity]]
            spread[name] = (
                float((inner.real.max(axis=0) - inner.real.min(axis=0)).max()),
                float((inner.imag.max(axis=0) - inner.imag.min(axis=0)).max()),
            )
            n_seam += int((np.abs(centred) > 1e-9).any(axis=1).sum())
        between = _deviation(parity_mean["up"], parity_mean["down"], cfl)
        up_vs_down = (float(np.abs(between.real).max()), float(np.abs(between.imag).max()))
        growth = -at_cell.imag
        k_max = int(np.argmax(growth))
        crossings = _growth_boundary(alphas, growth)
        lines.append(
            f"{cfl:6.2f}  {d_re:.3e}  {d_im:.3e}  {spread['up'][0]:.3e}  {spread['up'][1]:.3e}  "
            f"{spread['down'][0]:.3e}  {spread['down'][1]:.3e}  {up_vs_down[0]:.3e}  "
            f"{up_vs_down[1]:.3e}  {n_seam:4d}  {table.diff[i, SAMPLE_CELL].max():.6e}  "
            f"{growth[k_max]:+.6e}  {alphas[k_max]:.6f}  "
            + ("none" if not crossings else " ".join(f"{a:.6f}" for a in crossings))
        )
    return lines, max_d_re, max_d_im


def _plot(table: _Table, reference: np.ndarray | None, scheme: int, tag: str) -> list[pathlib.Path]:
    """Figures 5 and 6 (his six CFLs, the exact curve, his table dashed) and 7 (theta = 0)."""
    if plt is None:
        return []
    alphas = np.asarray(ALPHA)
    written = []
    paper = [cfl for cfl in CFL_PAPER if any(math.isclose(cfl, c) for c in table.cfls)]
    if paper:
        for name, part, exact, ylabel in (
            ("fig5", lambda w: w.real, alphas, r"Re $\tilde\omega$"),
            ("fig6", lambda w: -w.imag, np.zeros_like(alphas), r"$-$Im $\tilde\omega$"),
        ):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(alphas, exact, color="black", linewidth=1.0, label="exact")
            for cfl in paper:
                i = next(j for j, c in enumerate(table.cfls) if math.isclose(c, cfl))
                (line,) = ax.plot(alphas, part(table.omega[i, SAMPLE_CELL]), label=f"CFL {cfl:g}")
                if reference is not None:
                    block = _fortran_block(reference, cfl)
                    ax.plot(
                        alphas,
                        part(block[:, 2] + 1j * block[:, 3]),
                        linestyle="--",
                        color=line.get_color(),
                        linewidth=0.8,
                    )
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"ihadv_tracer {scheme}, theta = 0 (icon4py solid, Fortran dashed, exact black)"
            )
            ax.legend(fontsize=8)
            ax.grid(True, linewidth=0.3)
            path = OUTPUT_DIR / f"{name}_{_SCHEME_IDS[scheme]}_theta0_{tag}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    if len(table.cfls) > 6:
        growth = -table.omega[:, SAMPLE_CELL, :].imag  # (cfl, alpha)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        contour = ax.contourf(alphas, table.cfls, growth, levels=21, cmap="RdBu_r")
        ax.contour(alphas, table.cfls, growth, levels=[0.0], colors="black", linewidths=1.0)
        fig.colorbar(contour, ax=ax, label=r"$-$Im $\tilde\omega$")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("CFL")
        ax.set_title(f"ihadv_tracer {scheme}, theta = 0: growth rate (black: zero)")
        path = OUTPUT_DIR / f"fig7_{_SCHEME_IDS[scheme]}_theta0_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


@pytest.mark.embedded_remap_error
@pytest.mark.skipif(not GRID_FILE.exists(), reason=f"Jocksch's grid file {GRID_FILE} not found")
@pytest.mark.parametrize("scheme", [2, 3], ids=_SCHEME_IDS.get)
@pytest.mark.parametrize("cfl_set", list(CFL_SETS))
def test_dispersion_relation_theta0(
    scheme: int,
    cfl_set: str,
    *,
    tmp_path: pathlib.Path,
    process_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend | None,
) -> None:
    cfls = CFL_SETS[cfl_set]
    backend_tag = data_alloc.backend_name(backend)
    tag = f"{cfl_set}_{backend_tag}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = wall_time.perf_counter()
    setup = _setup(scheme, backend, process_props, tmp_path)
    setup_seconds = wall_time.perf_counter() - start
    table = _dispersion(setup, cfls)
    assert np.isfinite(table.omega).all()

    # the levels are independent: one alpha on every level gives the same column
    alpha = ALPHA[INDEPENDENCE_LEVEL]
    re0, im0 = _initial_wave(setup, (alpha,) * NUM_LEVELS)
    dtime = _time_step(INDEPENDENCE_CFL)
    q_single = _advect(setup, re0, dtime) + 1j * _advect(setup, im0, dtime)
    i_cfl = next((i for i, c in enumerate(cfls) if math.isclose(c, INDEPENDENCE_CFL)), None)
    if i_cfl is not None:
        expected = table.q_new[i_cfl, :, INDEPENDENCE_LEVEL]
        np.testing.assert_array_equal(q_single, expected[:, None] * np.ones((1, NUM_LEVELS)))

    reference_path = REFERENCE_TABLES[scheme]
    reference = _read_fortran_table(reference_path) if reference_path.exists() else None

    stem = f"{_SCHEME_IDS[scheme]}_theta0"
    if cfl_set == "full":
        _write_fortran_format(OUTPUT_DIR / f"{stem}.txt", table, SAMPLE_CELL)
    _write_fortran_format(OUTPUT_DIR / f"{stem}_{tag}.txt", table, SAMPLE_CELL)
    np.savez(
        OUTPUT_DIR / f"{stem}_{tag}_cells.npz",
        cfl=np.asarray(cfls),
        alpha=np.asarray(ALPHA),
        omega=table.omega,
        diff=table.diff,
        q_new=table.q_new,
        tip_up=setup.tip_up,
        cell_x=setup.cell_x,
        cell_y=setup.cell_y,
    )
    lines, max_d_re, max_d_im = _summarise(table, setup, reference)
    total_seconds = wall_time.perf_counter() - start
    header = [
        f"# scheme {scheme}, theta = 0, backend {backend_tag}, cfl set {cfl_set} "
        f"({len(cfls)} CFL x {NUM_LEVELS} alpha), reference {reference_path}",
        f"# setup {setup_seconds:.1f} s, {table.seconds_per_run:.3f} s per Advection.run, "
        f"total {total_seconds:.1f} s",
        f"# max |Re omega - Fortran| {max_d_re:.3e}, max |Im omega - Fortran| {max_d_im:.3e}",
    ]
    summary = OUTPUT_DIR / f"{stem}_{tag}_summary.txt"
    summary.write_text("\n".join(header + lines) + "\n")
    figures = _plot(table, reference, scheme, tag)
    print("\n" + "\n".join(header + lines))
    print("figures: " + ", ".join(str(f) for f in figures))

    # the longest wave is carried the right way at about the right speed: its one-step
    # error is small against 2 sin(CFL alpha), the error of a wave standing still or
    # carried the wrong way (the sign of the phase against the sign of the wind)
    for i, cfl in enumerate(cfls):
        wrong_way = 2.0 * math.sin(cfl * ALPHA[1])
        assert table.diff[i, SAMPLE_CELL, 1] < TRANSLATION_TOLERANCE * wrong_way, (
            f"CFL {cfl}: diff {table.diff[i, SAMPLE_CELL, 1]:.3e} against {wrong_way:.3e}"
        )
    if reference is None:
        pytest.skip(f"Fortran reference table {reference_path} not available")
    tolerance_re, tolerance_im = FORTRAN_TOLERANCE[scheme]
    assert max_d_re <= tolerance_re
    assert max_d_im <= tolerance_im

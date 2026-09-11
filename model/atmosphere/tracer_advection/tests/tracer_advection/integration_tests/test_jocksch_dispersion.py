# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The dispersion relation of the linear and quadratic schemes (Jocksch et al., PPAM 2026, section 4).

Figures 5 and 6 of the paper (phase speed and growth rate of a plane wave against its
wavenumber for CFL 0.01, 0.1, ..., 0.5) and figure 7 (growth rate over the (CFL,
wavenumber) plane) at theta = 0 and 30 degrees, for the schemes 2 (Miura, linear
reconstruction) and 3 (Miura, quadratic), unlimited: the limiters are nonlinear and the
real and imaginary parts of the wave are two real tracers whose responses must superpose.

The setup is A. Jocksch's pair of dispersion blocks in src/atm_dyn_iconam/mo_nh_stepping.f90
of icon-exclaim branch transport_ajocksch (pristine commit dacecf46aa: theta = 0 at lines
3322-3466, theta = 30 degrees at 3467-3678). Line numbers ':nnnn' below are of the capture
branch transport_ajocksch_capture at commit db7a1f149d (worktree icon-ajocksch of the
tracer-advection port workspace), which runs both blocks unchanged behind the
ICON_DISPERSION switch and produced the Fortran tables of the port. The theta = 0 block
(:3397-3550) produced his tables dispersion_linear.txt (scheme 2) and
dispersion_quadratic.txt (scheme 3):

- his grid torus_grid_r4_c200_elen100.nc (880 cells, 5 km edges, centred coordinates), a
  uniform wind |v| = 1 along x (angle 0 at :3398-3400), the edge mass flux ``vn_traj =
  mass_flx_me = u n_x`` (his edge-vector formula :3401-3466 equals the edge normal on his
  grid, every ``n_x >= 0``), air mass 1 and no vertical flux (:3467-3472);
- CFL = icfl / 100 for icfl = 0, 2, ..., 100 with 0 -> 0.01 (the default list :3147-3153,
  the loop :3484-3485); wavenumber alpha = ireal / 100 pi for ireal = 0, 2, ..., 100
  (:3486-3487); the wave ``exp(-i x 2 / a alpha)`` on the cell centres as tracers 1
  (real) and 2 (imaginary), :3488-3495, set fresh for every (CFL, alpha);
- one step_advection with ``dt = (a / 2) 1.9999999999 CFL`` (:3498), so CFL is the Courant
  number with respect to the edge length ``a`` (the wind covers ``2 CFL`` half-edges);
- ``omega = -ln(q_new / q_now) i / dt (a / 2)`` at the Fortran cell 883/3 = 294 (:3528,
  1-based; cell index 293 here) and ``diff = |q_new - q_now / exp(-i 2 CFL alpha)|``
  (:3534-3543); ``print '(F20.16, F20.16, F20.16, F20.16)'`` of CFL, alpha, Re omega,
  Im omega (:3544; the fifth item, diff, wraps onto a second line under this format in
  his ``_new`` tables, and is absent from dispersion_linear/quadratic.txt); a blank line
  per CFL block (:3546).
  The older variant of the block (dispersion_relation/MO_NH_STEPPING/
  mo_nh_stepping.f90_dispersion:2158-2183) samples cell 880/3 = 293, normalises by cell
  291 and prints ``Re omega + 2 pi``; his tables carry the raw principal value (CFL 1,
  alpha = pi: -1e-16), so it is the live block that is mirrored here.

Here one alpha per vertical level (his own device in the theta = 30 degree block
:3551-3791; the columns of the granule are independent, which the test asserts by
re-running one CFL with a single alpha on every level), so the 51 x 51 table costs 51 CFL
x 2 tracers ``Advection.run`` calls per scheme. The granule is the one the driver builds on
his grid file, with the states of the cylinder experiment (jocksch_cylinder.yaml) and the
metrics of a flat 51-level column; ``grf_tend_tracer`` is zero here, -1 in the Fortran
(:3473, unused on a torus). Outputs (git-ignored, under weno_data/dispersion of the
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

Theta = 30 degrees, his chequerboard block (:3551-3791), whose tables are
linear_oblique.txt_new (scheme 2) and quadratic_oblique5.txt (scheme 3), 51 CFL x 59 alpha:

- the wind at ``angle = dble(30)/180d0*acos(-1d0)`` (:3567-3572) as his edge-vector mass
  flux ``-y_comp/length u_x + x_comp/length u_y`` on every level (:3573-3643: edge vector
  vertex 2 - vertex 1 with the periodic-wrap corrections, divided by ``length_ref``, the
  length of edge 695), here computed from the grid's vertex coordinates and E2V (asserted
  equal to ``u . n`` of the geometry within 1e-12, which fixes the vertex order);
- alpha = ireal / 100 pi for ireal = 0, 2, ..., 116 on level 1 + ireal / 2 (59 levels),
  the wave ``exp(-i alpha (x u_x + y u_y) 2 / a)`` set once per CFL (:3651-3672);
- 1000 iterations (:3678) of {one step (dt :3679, step_advection :3680-3700); the
  frequency at cell k = 695 and its change against the previous iteration, the capture's
  ``#conv`` record (:3703-3713: ``n_last`` the last iteration with ``|d omega| >= 1e-12``
  (1 at the first), ``resid`` the change after the last iteration); unless it is the last
  iteration, the wave re-imposed on tracer(n_now) from the advected field (:3714-3753):
  every cell j gets ``tracer_ref(mod(j, 2) + 1) exp(-i ((x_j - x_ref) u_x + (y_j - y_ref)
  u_y) 2 / a alpha) / |tracer_ref(1)|``, reference 2 the cell k = 695, reference 1 its
  neighbour 696 (here 0-based 694 and 695; on his grid the index parity is the triangle
  orientation, 695 tip-down and 696 tip-up, asserted)};
- after the last iteration ``omega`` at 695 and ``omega2`` at 696 from the last imposed
  wave and the last advected field (:3756-3779), ``print '(6F25.16)'`` with a blank line
  per CFL block (:3780-3783) and the ``#conv`` lines ``(a, F25.16, F25.16, i8, es14.3)``
  (:3784-3787). Written here in the same layout (his files have no blank line after the
  last block).

The per-CFL results are saved as soon as a CFL is done (``theta30_parts/``), so the full
table runs as a chain of 30-minute jobs (ICON4PY_DISPERSION_THETA30_CFLS, a deadline in
ICON4PY_DISPERSION_DEADLINE, done CFLs are skipped); the gated sets are the paper's six
CFLs and the stability pair 0.42 / 0.44 (validation level; 3.7 s / 9.4 s per CFL for
scheme 2 / 3 on gtfn_cpu on a compute node, the full tables 3.2 min (192.5 s) / 8.0 min).
The measured numbers are in the status note (docs/weno_idealized_status.md).
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import time as wall_time
from typing import Any, Final

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
    from matplotlib import colors as mpl_colors
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

#: our build's theta = 0 table (W4b, the same block; for the icon4py-vs-build comparison)
FORTRAN_BUILD_TABLES: Final[dict[int, pathlib.Path]] = {
    scheme: WORKSPACE
    / "weno_data"
    / "reference"
    / "jocksch_grid"
    / f"dispersion_ihadv{scheme}_theta0_full"
    / "dispersion.txt"
    for scheme in (2, 3)
}

#: the edge length a of his grid; 'length' in the Fortran (the reference edge 695, :3401-3415)
EDGE_LENGTH: Final = 5000.0
#: dt = 1.0 * (length / 2) * 1.9999999999 * cfl, mo_nh_stepping.f90:3498 (:3679 at theta = 30)
DT_FACTOR: Final = 1.9999999999
#: cfl = icfl / 100, icfl = 0, 2, ..., 100, 0 -> 0.01 (:3147-3153)
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
#: alpha = ireal / 100 * acos(-1), ireal = 0, 2, ..., 100 (:3487), one per level
ALPHA: Final[tuple[float, ...]] = tuple(ireal / 100 * math.pi for ireal in range(0, 101, 2))
NUM_LEVELS: Final = len(ALPHA)
#: his sampling cell 883/3 = 294 (1-based, :3528); it is a tip-up cell at (-17.5, 14.43) km
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
#: the single-CFL set on another backend against the gtfn_cpu full table at his cell
#: (max |d Re omega|, |d Im omega| against the F20.16 print; measured dace_cpu: scheme 2
#: 5.6e-17 / 5.6e-17, scheme 3 4.4e-16 / 1.1e-15, the tracer within 2.5e-16 amplified by
#: the logarithm; measured on dace_cpu only, untested on the GPU backends)
BACKEND_TOLERANCE: Final = 3e-15
BACKEND_REFERENCE_TAG: Final = "full_run_gtfn_cpu"

_SCHEME_IDS: Final = {2: "ihadv2", 3: "ihadv3"}

# --- theta = 30 degrees (the chequerboard block :3551-3791) ---
#: angle = dble(iangle) / 180d0 * acos(-1d0), iangle = 30 (:3567-3568)
WIND_ANGLE_30: Final = 30 / 180 * math.acos(-1.0)
#: alpha = ireal / 100 * acos(-1), ireal = 0, 2, ..., 116 on level 1 + ireal / 2 (:3569-3570)
ALPHA_30: Final[tuple[float, ...]] = tuple(ireal / 100 * math.pi for ireal in range(0, 117, 2))
NUM_LEVELS_30: Final = len(ALPHA_30)
#: the Fortran's k = 695 (1-based, :3721): reference 2 of the re-imposition and the cell of
#: omega; reference 1 and omega2 are at k + 1 = 696 (:3722-3723)
REFERENCE_CELL_30: Final = 695 - 1
#: the Fortran's reference edge k = 695 of length_ref (:3575, :3581-3589)
REFERENCE_EDGE_30: Final = 695 - 1
#: do iodd_even = n_iter, 1, -1 with n_iter = 1000 (:3156, :3678)
NUM_ITERATIONS_30: Final = 1000
#: the #conv record's threshold on |omega - omega_prev| (:3711)
CONVERGENCE_THRESHOLD_30: Final = 1e-12
#: a row counts as converged for the comparisons if its last change is at most this
#: (the capture notes' and compare_dispersion.py's default cut)
CONVERGED_RESID_30: Final = 1e-8
#: growth means Im omega < -GROWTH_TOLERANCE at either reference cell (ignores the
#: +-5e-16 of the alpha = 0 rows)
GROWTH_TOLERANCE: Final = 1e-14
CFL_STABILITY: Final[tuple[float, ...]] = (0.42, 0.44)
CFL_SETS_30: Final[dict[str, tuple[float, ...] | None]] = {
    "paper": CFL_PAPER,
    "stability": CFL_STABILITY,
    #: the job scripts' lists (ICON4PY_DISPERSION_THETA30_CFLS), e.g. the full CFL_FULL
    "env": None,
}
#: his theta = 30 tables, (6F25.16) cfl, alpha, Re/Im omega at 695, Re/Im omega at 696
REFERENCE_TABLES_30: Final[dict[int, pathlib.Path]] = {
    2: pathlib.Path(
        "/capstor/scratch/cscs/ajocksch/dispersion_relation/icon/run/linear_oblique.txt_new"
    ),
    3: pathlib.Path(
        "/capstor/scratch/cscs/ajocksch/dispersion_relation/icon/run/quadratic_oblique5.txt"
    ),
}
#: our build's full theta = 30 tables (W4c-F) and their #conv records
FORTRAN_BUILD_DIRS_30: Final[dict[int, pathlib.Path]] = {
    scheme: WORKSPACE
    / "weno_data"
    / "reference"
    / "jocksch_grid"
    / f"dispersion_ihadv{scheme}_theta30_full"
    for scheme in (2, 3)
}
#: max |d omega| (the four columns) against his table over the rows converged here
#: (resid <= CONVERGED_RESID_30) and all CFLs of the 'paper' or the 'stability' set, per
#: scheme; measured on gtfn_cpu only. The paper-set maximum is the CFL 0.01 block, so the
#: gate is 3x that and 17x (scheme 2) / 22x (scheme 3) the CFL 0.5 block; the stability set
#: measures 4.4e-15 / 3.0e-15 (CFL 0.44)
FORTRAN_TOLERANCE_30: Final[dict[int, float]] = {
    2: 6e-14,  # 1.9e-14 (CFL 0.01; 2.0e-15 .. 3.5e-15 at 0.1 .. 0.5)
    3: 6e-14,  # 1.8e-14 (CFL 0.01; 2.2e-15 .. 2.7e-15 at 0.1 .. 0.5, 2.66e-15 at 0.5)
}
PARTS_DIR: Final = OUTPUT_DIR / "theta30_parts"


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


_setups: dict[tuple[int, str, int], _Setup] = {}


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
    theta: int = 0,
) -> _Setup:
    """The granule the driver builds on his grid, the wind and the air mass of his block.

    ``theta`` 0 or 30 (degrees): the level count (51 / 59 alphas) and the wind of the block.
    """
    assert theta in (0, 30)
    key = (scheme, data_alloc.backend_name(backend), theta)
    if key in _setups:
        return _setups[key]
    config = config_io.read_yaml_str(
        EXPERIMENT_CONFIG.read_text(), driver_config.ExperimentConfig
    ).with_overrides(
        vertical_grid={"num_levels": NUM_LEVELS if theta == 0 else NUM_LEVELS_30},
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
    if theta == 0:
        # angle 0: u_x = 1, u_y = 0 (:3398-3400); vn_traj = mass_flx_me, mass_flx_ic = 0 (:3467-3469)
        lin_hor_adv_ic.prescribe_uniform_wind(
            prep_adv_state=prep_adv,
            primal_normal_x=normal_x,
            primal_normal_y=normal_y,
            u=1.0,
            v=0.0,
        )
    else:
        # his edge-vector flux on every level (:3573-3641), vn_traj = mass_flx_me and
        # mass_flx_ic = 0 (:3643-3644)
        flux = _edge_vector_mass_flux(
            vertex_x=geometry.get(geometry_attrs.VERTEX_X).asnumpy(),
            vertex_y=geometry.get(geometry_attrs.VERTEX_Y).asnumpy(),
            e2v=grid.get_connectivity("E2V").asnumpy(),
            wind_angle=WIND_ANGLE_30,
        )
        u_dot_n = math.cos(WIND_ANGLE_30) * data_alloc.as_numpy(normal_x) + math.sin(
            WIND_ANGLE_30
        ) * data_alloc.as_numpy(normal_y)
        # the same wind as the geometry's normals (and so the same vertex order); measured
        # 8.9e-16, a reversed vertex order gives O(1)
        print(f"theta = 30 wind: max |edge-vector flux - u.n| {np.abs(flux - u_dot_n).max():.3e}")
        np.testing.assert_allclose(flux, u_dot_n, rtol=0.0, atol=1e-12)
        xp = data_alloc.array_namespace(prep_adv.mass_flx_me.ndarray)
        prep_adv.mass_flx_me.ndarray[:, :] = xp.asarray(flux)[:, None]
        prep_adv.vn_traj.ndarray[:, :] = xp.asarray(flux)[:, None]
        prep_adv.mass_flx_ic.ndarray[:, :] = 0.0
    # airmass_now = airmass_new = 1 (:3471-3472, :3646-3647); grf_tend_tracer is unused on a torus
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
    tip_up = _classify_tip_up(cell_y)
    if theta == 30:
        # the chequerboard's index parity (mod(j, 2), :3739) is the triangle orientation
        assert (tip_up == (np.arange(cell_y.size) % 2 == 1)).all()
    _setups[key] = _Setup(
        advection=icon4py_driver.granules.tracer_advection,
        grid=grid,
        backend=backend,
        cell_x=cell_x,
        cell_y=cell_y,
        tip_up=tip_up,
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


def _or_na(value: float) -> str:
    """``value`` as ``.3e``, or "n/a" for NaN (a comparison whose table is missing)."""
    return "n/a" if math.isnan(value) else f"{value:.3e}"


def _time_step(cfl: float) -> float:
    return 1.0 * (EDGE_LENGTH / 2) * DT_FACTOR * cfl


def _initial_wave(
    setup: _Setup, alphas: tuple[float, ...], wind_angle: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary tracers (cells, levels), level k carrying the wave of alphas[k]."""
    re = np.empty((setup.cell_x.size, len(alphas)))
    im = np.empty_like(re)
    for k, alpha in enumerate(alphas):
        re[:, k], im[:, k] = plane_wave.plane_wave(
            alpha=alpha,
            wind_angle=wind_angle,
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
    """(rows, 4 or 6) of a table, the blank block separators, '#' lines and ``diff`` lines dropped.

    Our build's theta = 0 tables carry the fifth printed item, ``diff``, on a line of its
    own (the four-descriptor format, :3544); only the rows of the table's column count
    (that of its widest row) are kept.
    """
    rows = [
        line.split()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    width = max(len(row) for row in rows)
    return np.asarray([[float(item) for item in row] for row in rows if len(row) == width])


def _fortran_block(table: np.ndarray, cfl: float, alphas: tuple[float, ...] = ALPHA) -> np.ndarray:
    """The rows of one CFL of his table, in his alpha order (asserted to be ``alphas``)."""
    block = table[np.isclose(table[:, 0], cfl, rtol=0.0, atol=1e-15)]
    assert block.shape[0] == len(alphas), f"CFL {cfl}: {block.shape[0]} rows in the reference"
    np.testing.assert_allclose(block[:, 1], alphas, rtol=0.0, atol=1e-15)
    return block


def _normalised_difference(cfl: float, d_omega: np.ndarray, im_omega: np.ndarray) -> np.ndarray:
    """``|d omega| 2 CFL |q_new / q_now|``: a frequency difference as the round-off of the ratio.

    ``omega = -ln(q_new / q_now) i / (1.9999999999 CFL)``, so a perturbation of the ratio by
    ``e`` moves omega by ``e / (1.9999999999 CFL |q_new / q_now|)``;
    ``|q_new / q_now| = exp(-1.9999999999 CFL Im omega)``.
    """
    return np.abs(d_omega) * 2.0 * cfl * np.exp(-DT_FACTOR * cfl * im_omega)


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
    table: _Table, setup: _Setup, reference: np.ndarray | None, build: np.ndarray | None = None
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
        "zero crossings; against our Fortran build's table (W4b, the same block): max "
        "|d Re omega|, |d Im omega| and the normalised difference |d omega| 2 CFL "
        "|q_new / q_now| (the round-off of the ratio; component max and modulus, max over "
        "alpha, |q_new / q_now| from the build's Im omega)",
        "# cfl  dRe  dIm  spread_up_re  spread_up_im  spread_down_re  spread_down_im  "
        "up_vs_down_re  up_vs_down_im  n_seam  diff_max  growth_max  alpha_at_growth_max  "
        "growth_zero_alphas  build_dRe  build_dIm  build_norm_component  build_norm_modulus",
    ]
    # NaN without his table, so that the summary prints n/a rather than 0
    max_d_re = max_d_im = 0.0 if reference is not None else math.nan
    build_norm_max: dict[str, list[float]] = {"all": [0.0, 0.0], "cfl<=0.5": [0.0, 0.0]}
    for i, cfl in enumerate(table.cfls):
        omega = table.omega[i]  # (cells, alpha)
        at_cell = omega[SAMPLE_CELL]
        if reference is not None:
            block = _fortran_block(reference, cfl)
            d_re = float(np.abs(at_cell.real - block[:, 2]).max())
            d_im = float(np.abs(at_cell.imag - block[:, 3]).max())
            max_d_re = max(max_d_re, d_re)
            max_d_im = max(max_d_im, d_im)
        else:
            d_re = d_im = math.nan
        build_columns = ""
        if build is not None:
            build_block = _fortran_block(build, cfl)
            b_re = np.abs(at_cell.real - build_block[:, 2])
            b_im = np.abs(at_cell.imag - build_block[:, 3])
            norm_component = _normalised_difference(cfl, np.maximum(b_re, b_im), build_block[:, 3])
            norm_modulus = _normalised_difference(cfl, np.hypot(b_re, b_im), build_block[:, 3])
            for key in ("all", "cfl<=0.5") if cfl <= 0.5 else ("all",):
                build_norm_max[key][0] = max(build_norm_max[key][0], float(norm_component.max()))
                build_norm_max[key][1] = max(build_norm_max[key][1], float(norm_modulus.max()))
            build_columns = (
                f"  {b_re.max():.3e}  {b_im.max():.3e}  {norm_component.max():.3e}  "
                f"{norm_modulus.max():.3e}"
            )

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
            + build_columns
        )
    if build is not None:
        lines.append(
            "# against our build, max normalised |d omega| 2 CFL |q_new / q_now| (component, "
            "modulus): all CFL "
            f"{build_norm_max['all'][0]:.3e}, {build_norm_max['all'][1]:.3e}; CFL <= 0.5 "
            f"{build_norm_max['cfl<=0.5'][0]:.3e}, {build_norm_max['cfl<=0.5'][1]:.3e}"
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
        _growth_panel(
            fig=fig,
            ax=ax,
            alphas=alphas,
            cfls=np.asarray(table.cfls),
            growth=growth,
            title=f"ihadv_tracer {scheme}, theta = 0: growth rate (black: zero)",
        )
        path = OUTPUT_DIR / f"fig7_{_SCHEME_IDS[scheme]}_theta0_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _growth_panel(
    *, fig: Any, ax: Any, alphas: np.ndarray, cfls: np.ndarray, growth: np.ndarray, title: str
) -> None:
    """Figure 7's panel: -Im omega over (alpha, CFL), a diverging map centred at zero.

    Growth (> 0) red, damping blue, each side scaled to its own extreme so that the weak
    growth near the stability limit stays visible; the zero contour black; NaN (rows that
    did not converge) left grey.
    """
    finite = growth[np.isfinite(growth)]
    scale = float(np.abs(finite).max()) if finite.size else 1.0
    tiny = 1e-12 * max(scale, 1e-300)
    vmin = min(float(finite.min()), -tiny)
    vmax = max(float(finite.max()), tiny)
    levels = np.concatenate([np.linspace(vmin, 0.0, 11)[:-1], np.linspace(0.0, vmax, 11)])
    norm = mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    ax.set_facecolor("0.8")
    contour = ax.contourf(alphas, cfls, growth, levels=levels, cmap="RdBu_r", norm=norm)
    if vmin < 0.0 < float(finite.max()):
        ax.contour(alphas, cfls, growth, levels=[0.0], colors="black", linewidths=1.0)
    fig.colorbar(contour, ax=ax, label=r"$-$Im $\tilde\omega$ (centred at 0)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("CFL")
    ax.set_title(title, fontsize=9)


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
    build_path = FORTRAN_BUILD_TABLES[scheme]
    build = _read_fortran_table(build_path) if build_path.exists() else None

    stem = f"{_SCHEME_IDS[scheme]}_theta0"
    # the saved gtfn_cpu full table, read before this run may rewrite it
    backend_reference_path = OUTPUT_DIR / f"{stem}_{BACKEND_REFERENCE_TAG}.txt"
    backend_reference = (
        _read_fortran_table(backend_reference_path)
        if cfl_set == "single" and backend_reference_path.exists()
        else None
    )
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
    lines, max_d_re, max_d_im = _summarise(table, setup, reference, build)
    total_seconds = wall_time.perf_counter() - start
    header = [
        f"# scheme {scheme}, theta = 0, backend {backend_tag}, cfl set {cfl_set} "
        f"({len(cfls)} CFL x {NUM_LEVELS} alpha), reference {reference_path}, our build {build_path}",
        f"# setup {setup_seconds:.1f} s, {table.seconds_per_run:.3f} s per Advection.run, "
        f"total {total_seconds:.1f} s",
        f"# max |Re omega - Fortran| {_or_na(max_d_re)}, max |Im omega - Fortran| {_or_na(max_d_im)}",
    ]
    backend_d = (math.nan, math.nan)
    if backend_reference is not None:
        backend_block = _fortran_block(backend_reference, CFL_SINGLE[0])
        at_cell = table.omega[0, SAMPLE_CELL]
        backend_d = (
            float(np.abs(at_cell.real - backend_block[:, 2]).max()),
            float(np.abs(at_cell.imag - backend_block[:, 3]).max()),
        )
        header.append(
            f"# against the saved {BACKEND_REFERENCE_TAG} table ({backend_reference_path.name}), "
            f"CFL {CFL_SINGLE[0]}: max |d Re omega| {backend_d[0]:.3e}, max |d Im omega| {backend_d[1]:.3e}"
        )
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
    if cfl_set == "single":
        # the second backend gives gtfn_cpu's table at his cell
        if backend_reference is None:
            pytest.skip(
                f"saved table {backend_reference_path} not available (run the full set first)"
            )
        assert max(backend_d) <= BACKEND_TOLERANCE, backend_d


# --------------------------------------------------------------------------------------
# theta = 30 degrees: the chequerboard iteration
# --------------------------------------------------------------------------------------


def _edge_vector_mass_flux(
    *, vertex_x: np.ndarray, vertex_y: np.ndarray, e2v: np.ndarray, wind_angle: float
) -> np.ndarray:
    """His edge mass flux for a uniform unit wind (mo_nh_stepping.f90:3573-3641).

    The edge vector ``vertex 2 - vertex 1``; an edge whose vector spans the periodic domain
    is replaced by the neighbouring-image vector (``+-length_ref`` for the horizontal edges,
    ``+-length_ref / 2``, ``+-length_ref sqrt(3) / 2`` for the others; the ``0.1`` and ``1.1``
    of the tests are default-real literals, :3612-3634); the flux is
    ``-y_comp / length_ref u_x + x_comp / length_ref u_y`` (:3635-3637), the same on every
    level. ``length_ref`` is the length of edge 695 (:3581-3589), asserted to be EDGE_LENGTH.
    """
    k = REFERENCE_EDGE_30
    length_ref = np.sqrt(
        (vertex_x[e2v[k, 0]] - vertex_x[e2v[k, 1]]) ** 2
        + (vertex_y[e2v[k, 0]] - vertex_y[e2v[k, 1]]) ** 2
    )
    assert length_ref == EDGE_LENGTH, length_ref
    x_comp = vertex_x[e2v[:, 1]] - vertex_x[e2v[:, 0]]
    y_comp = vertex_y[e2v[:, 1]] - vertex_y[e2v[:, 0]]
    tenth = length_ref * float(np.float32(0.1))
    beyond = length_ref * float(np.float32(1.1))
    horizontal = np.abs(y_comp) < tenth
    x_new = x_comp.copy()
    y_new = y_comp.copy()
    # (each assignment leaves the value outside the next test's range, so the tests can
    # read the original components)
    x_new[horizontal & (x_comp > beyond)] = -length_ref
    y_new[horizontal & (x_comp > beyond)] = 0.0
    x_new[horizontal & (x_comp < -beyond)] = length_ref
    y_new[horizontal & (x_comp < -beyond)] = 0.0
    x_new[~horizontal & (x_comp > beyond)] = -length_ref * 0.5
    x_new[~horizontal & (x_comp < -beyond)] = length_ref * 0.5
    y_new[~horizontal & (y_comp > beyond)] = -length_ref * math.sqrt(3.0) * 0.5
    y_new[~horizontal & (y_comp < -beyond)] = length_ref * math.sqrt(3.0) * 0.5
    u_x = math.cos(wind_angle)
    u_y = math.sin(wind_angle)
    return -(y_new / length_ref * u_x) + x_new / length_ref * u_y


def _complex(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """``cmplx(re, im)`` exactly (no arithmetic on the real part, signed zeros kept)."""
    out = np.empty(re.shape, dtype=complex)
    out.real = re
    out.imag = im
    return out


@dataclasses.dataclass(frozen=True)
class _Theta30Block:
    """One CFL of the theta = 30 degree table (the Fortran cells 695 and 696)."""

    cfl: float
    #: (2, alpha) omega at the Fortran cell 695 and omega2 at 696 after the last iteration
    omega: np.ndarray
    #: (alpha,) the #conv record at 695: the last iteration with |d omega| >= 1e-12, and
    #: |omega - omega_prev| after the last iteration
    n_last: np.ndarray
    resid: np.ndarray
    #: (alpha,) the first iteration (>= 2) with |d omega| < 1e-12, 0 if none, and (2, alpha)
    #: omega / omega2 at that iteration: the value a per-level early stop would give
    n_first: np.ndarray
    omega_first: np.ndarray
    #: (cells, alpha) the last imposed wave and the last advected field
    q_now: np.ndarray
    q_new: np.ndarray
    seconds: float
    seconds_advection: float
    iterations: int


def _chequerboard_iteration(
    setup: _Setup, cfl: float, num_iterations: int = NUM_ITERATIONS_30
) -> _Theta30Block:
    """His iteration for one CFL on all 59 levels (mo_nh_stepping.f90:3561-3787)."""
    start = wall_time.perf_counter()
    advection_seconds = 0.0
    dtime = _time_step(cfl)  # :3679
    cells = [REFERENCE_CELL_30, REFERENCE_CELL_30 + 1]
    phase_factor = plane_wave.chequerboard_phase_factor(
        alpha=np.asarray(ALPHA_30),
        wind_angle=WIND_ANGLE_30,
        cell_center_x=setup.cell_x,
        cell_center_y=setup.cell_y,
        reference_cell=REFERENCE_CELL_30,
        edge_length=EDGE_LENGTH,
    )
    now_re, now_im, new_re, new_im = (
        data_alloc.zero_field(setup.grid, dims.CellDim, dims.KDim, allocator=setup.backend)
        for _ in range(4)
    )
    xp = data_alloc.array_namespace(now_re.ndarray)
    re, im = _initial_wave(setup, ALPHA_30, WIND_ANGLE_30)  # :3651-3672
    now_re.ndarray[...] = xp.asarray(re)
    now_im.ndarray[...] = xp.asarray(im)

    levels = NUM_LEVELS_30
    omega_prev = np.zeros(levels, dtype=complex)  # :3564-3566
    n_last = np.zeros(levels, dtype=np.int64)
    resid = np.zeros(levels)
    n_first = np.zeros(levels, dtype=np.int64)
    omega_first = np.full((2, levels), complex(math.nan, math.nan))
    q_now = q_new = np.empty(0)
    for it in range(1, num_iterations + 1):  # it = n_iter - iodd_even + 1 (:3678, :3703)
        tick = wall_time.perf_counter()
        for now, new in ((now_re, new_re), (now_im, new_im)):
            new.ndarray[...] = 0.0
            setup.advection.run(
                diagnostic_state=setup.diagnostic_state,
                prep_adv=setup.prep_adv,
                p_tracer_now=now,
                p_tracer_new=new,
                dtime=dtime,
            )
        advection_seconds += wall_time.perf_counter() - tick
        q_new = _complex(new_re.asnumpy(), new_im.asnumpy())
        q_now = _complex(now_re.asnumpy(), now_im.asnumpy())
        # convergence record at cell 695 (:3704-3713)
        omega_it = plane_wave.numerical_frequency(
            q_now=q_now[REFERENCE_CELL_30],
            q_new=q_new[REFERENCE_CELL_30],
            dtime=dtime,
            edge_length=EDGE_LENGTH,
        )
        resid = np.abs(omega_it - omega_prev)
        if it == 1:
            n_last[:] = 1
        else:
            n_last[~(resid < CONVERGENCE_THRESHOLD_30)] = it
            newly = (n_first == 0) & (resid < CONVERGENCE_THRESHOLD_30)
            if newly.any():
                n_first[newly] = it
                omega_first[:, newly] = plane_wave.numerical_frequency(
                    q_now=q_now[cells][:, newly],
                    q_new=q_new[cells][:, newly],
                    dtime=dtime,
                    edge_length=EDGE_LENGTH,
                )
        omega_prev = omega_it
        if it < num_iterations:  # iodd_even > 1 (:3718): re-impose (:3731-3751)
            re, im = plane_wave.reimpose_chequerboard_wave(
                q_new=q_new, phase_factor=phase_factor, reference_cell=REFERENCE_CELL_30
            )
            now_re.ndarray[...] = xp.asarray(re)
            now_im.ndarray[...] = xp.asarray(im)
    # omega at 695 and omega2 at 696 (:3770-3779)
    omega = plane_wave.numerical_frequency(
        q_now=q_now[cells], q_new=q_new[cells], dtime=dtime, edge_length=EDGE_LENGTH
    )
    return _Theta30Block(
        cfl=cfl,
        omega=omega,
        n_last=n_last,
        resid=resid,
        n_first=n_first,
        omega_first=omega_first,
        q_now=q_now,
        q_new=q_new,
        seconds=wall_time.perf_counter() - start,
        seconds_advection=advection_seconds,
        iterations=num_iterations,
    )


def _part_path(parts_dir: pathlib.Path, scheme: int, cfl: float) -> pathlib.Path:
    return parts_dir / f"{_SCHEME_IDS[scheme]}_cfl{cfl:.2f}.npz"


def _save_part(path: pathlib.Path, block: _Theta30Block) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{f.name: getattr(block, f.name) for f in dataclasses.fields(block)})
    tmp.replace(path)


def _load_part(path: pathlib.Path) -> _Theta30Block:
    with np.load(path) as data:
        values = {f.name: data[f.name] for f in dataclasses.fields(_Theta30Block)}
    for name in ("cfl", "seconds", "seconds_advection"):
        values[name] = float(values[name])
    values["iterations"] = int(values["iterations"])
    return _Theta30Block(**values)


def _write_theta30_table(path: pathlib.Path, blocks: list[_Theta30Block]) -> None:
    """His layout: (6F25.16) per alpha (:3780-3781), a blank line between CFL blocks."""
    lines = []
    for i, block in enumerate(blocks):
        if i:
            lines.append(" ")
        for k, alpha in enumerate(ALPHA_30):
            w1, w2 = block.omega[0, k], block.omega[1, k]
            lines.append(
                f"{block.cfl:25.16f}{alpha:25.16f}{w1.real:25.16f}{w1.imag:25.16f}"
                f"{w2.real:25.16f}{w2.imag:25.16f}"
            )
    path.write_text("\n".join(lines) + "\n")


def _write_theta30_conv(path: pathlib.Path, blocks: list[_Theta30Block], backend_tag: str) -> None:
    """The capture's #conv lines (a, F25.16, F25.16, i8, es14.3) and the per-CFL wall time (:3784-3787)."""
    lines = [
        "#conv columns: cfl, alpha, last iteration with |omega - omega_prev| >= 1e-12, "
        "|omega - omega_prev| after the last iteration (cell 695)",
        f"#icon4py backend {backend_tag}; per CFL also: first iteration with |d omega| < 1e-12 "
        "(0: never) as '#first cfl alpha n_first'",
    ]
    for block in blocks:
        for k, alpha in enumerate(ALPHA_30):
            lines.append(
                f"#conv{block.cfl:25.16f}{alpha:25.16f}{int(block.n_last[k]):8d}"
                f"{float(block.resid[k]):14.3E}"
            )
        for k, alpha in enumerate(ALPHA_30):
            lines.append(f"#first{block.cfl:25.16f}{alpha:25.16f}{int(block.n_first[k]):8d}")
        lines.append(
            f"#dispersion theta30 cfl={block.cfl:25.16f} wall={block.seconds:12.3f} s "
            f"(Advection.run {block.seconds_advection:.3f} s, {block.iterations} iterations)"
        )
    path.write_text("\n".join(lines) + "\n")


def _read_conv(path: pathlib.Path) -> dict[tuple[float, float], tuple[int, float]]:
    """(cfl, alpha) rounded to 12 digits -> (n_last, resid) from #conv lines."""
    records = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "#conv":
            try:
                key = (round(float(parts[1]), 12), round(float(parts[2]), 12))
                records[key] = (int(parts[3]), float(parts[4]))
            except ValueError:
                continue
    return records


@dataclasses.dataclass(frozen=True)
class _Theta30Stats:
    """The per-CFL comparison and statistics of one theta = 30 table."""

    lines: list[str]
    #: max |d omega| over the four columns against his table on the converged rows / all rows
    #: (NaN without his table, likewise the build values without our build)
    his_converged: float
    his_all: float
    #: the same against our build (rows converged in both)
    build_converged: float
    build_all: float
    #: max normalised |d omega| 2 CFL |q_new / q_now| against our build, rows converged
    #: in both, CFL <= 0.5: component max, modulus; and on rows with resid < 1e-12 in both
    build_normalised: tuple[float, float]
    build_normalised_tight: tuple[float, float]
    #: CFL -> (min Im omega on converged rows, alpha, number of growing alphas, converged rows)
    stability: dict[float, tuple[float, float, int, int]]


def _theta30_statistics(
    blocks: list[_Theta30Block],
    reference: np.ndarray | None,
    build: np.ndarray | None,
    build_conv: dict[tuple[float, float], tuple[int, float]] | None,
) -> _Theta30Stats:
    alphas = np.asarray(ALPHA_30)
    lines = [
        "# per CFL (cells: the Fortran's 695 and 696; 'converged' = resid <= "
        f"{CONVERGED_RESID_30:g} at 695 after {blocks[0].iterations} iterations):",
        "#  n_conv: converged rows; n>1e-12 / n>1e-8 / n>1e-4: rows with resid above; "
        "med_n_last: median #conv iteration of the rows with resid < 1e-12",
        "#  his_conv / his_all: max |d| of the four omega columns against his table, converged / all rows",
        "#  build_conv / build_all: the same against our Fortran build (converged in both)",
        "#  norm_c / norm_m: max |d omega| 2 CFL |q_new / q_now| against our build (component max / "
        "modulus, both cells, converged in both; |q_new / q_now| from the build's Im omega)",
        "#  norm_c12: norm_c on rows with resid < 1e-12 in both",
        "#  min_im / at_alpha / n_grow: min Im omega over the converged rows and both cells, its alpha, "
        f"the number of converged alphas with Im omega < -{GROWTH_TOLERANCE:g} at either cell",
        "#  parity / parity_re: max |omega(695) - omega(696)| and |Re omega(695) - Re omega(696)| "
        "on the converged rows",
        "#  early: max |omega(n_first) - omega(last)| over both cells and the rows with resid < 1e-12 "
        "(what a per-level stop at |d omega| < 1e-12 would change), n_bitequal: such rows unchanged",
        "#  wall / adv: seconds for the CFL / in Advection.run",
        "# cfl  n_conv  n>1e-12  n>1e-8  n>1e-4  med_n_last  his_conv  his_all  build_conv  "
        "build_all  norm_c  norm_m  norm_c12  min_im  at_alpha  n_grow  parity  parity_re  "
        "early  n_bitequal  wall  adv",
    ]
    # NaN without his table / our build, so that the summary prints n/a rather than 0
    his_converged = his_all = 0.0 if reference is not None else math.nan
    build_converged = build_all = 0.0 if build is not None and build_conv is not None else math.nan
    normalised = [0.0, 0.0] if build is not None and build_conv is not None else [math.nan] * 2
    normalised_tight = list(normalised)
    stability = {}
    for block in blocks:
        cfl = block.cfl
        converged = block.resid <= CONVERGED_RESID_30
        tight = block.resid < CONVERGENCE_THRESHOLD_30
        columns = np.stack(
            [block.omega[0].real, block.omega[0].imag, block.omega[1].real, block.omega[1].imag],
            axis=1,
        )

        def _max(values: np.ndarray, mask: np.ndarray) -> float:
            return float(values[mask].max()) if mask.any() else math.nan

        d_his_c = d_his_a = d_build_c = d_build_a = math.nan
        norm_c = norm_m = norm_c12 = math.nan
        if reference is not None:
            d = np.abs(columns - _fortran_block(reference, cfl, ALPHA_30)[:, 2:6]).max(axis=1)
            d_his_c, d_his_a = _max(d, converged), float(d.max())
            his_converged = np.nanmax([his_converged, d_his_c])
            his_all = max(his_all, d_his_a)
        if build is not None and build_conv is not None:
            build_block = _fortran_block(build, cfl, ALPHA_30)
            resid_build = np.asarray(
                [build_conv[(round(cfl, 12), round(float(a), 12))][1] for a in alphas]
            )
            both = converged & (resid_build <= CONVERGED_RESID_30)
            both_tight = tight & (resid_build < CONVERGENCE_THRESHOLD_30)
            diff = columns - build_block[:, 2:6]
            d = np.abs(diff).max(axis=1)
            d_build_c, d_build_a = _max(d, both), float(d.max())
            build_converged = np.nanmax([build_converged, d_build_c])
            build_all = max(build_all, d_build_a)
            component = np.maximum(
                _normalised_difference(
                    cfl, np.maximum(np.abs(diff[:, 0]), np.abs(diff[:, 1])), build_block[:, 3]
                ),
                _normalised_difference(
                    cfl, np.maximum(np.abs(diff[:, 2]), np.abs(diff[:, 3])), build_block[:, 5]
                ),
            )
            modulus = np.maximum(
                _normalised_difference(cfl, np.hypot(diff[:, 0], diff[:, 1]), build_block[:, 3]),
                _normalised_difference(cfl, np.hypot(diff[:, 2], diff[:, 3]), build_block[:, 5]),
            )
            norm_c, norm_m, norm_c12 = (
                _max(component, both),
                _max(modulus, both),
                _max(component, both_tight),
            )
            if cfl <= 0.5:
                normalised[0] = np.nanmax([normalised[0], norm_c])
                normalised[1] = np.nanmax([normalised[1], norm_m])
                normalised_tight[0] = np.nanmax([normalised_tight[0], norm_c12])
                normalised_tight[1] = np.nanmax([normalised_tight[1], _max(modulus, both_tight)])
        im_min_cells = np.minimum(block.omega[0].imag, block.omega[1].imag)
        if converged.any():
            k_min = int(np.flatnonzero(converged)[np.argmin(im_min_cells[converged])])
            min_im, at_alpha = float(im_min_cells[k_min]), float(alphas[k_min])
        else:
            min_im, at_alpha = math.nan, math.nan
        n_grow = int((converged & (im_min_cells < -GROWTH_TOLERANCE)).sum())
        stability[cfl] = (min_im, at_alpha, n_grow, int(converged.sum()))
        parity = _max(np.abs(block.omega[0] - block.omega[1]), converged)
        parity_re = _max(np.abs(block.omega[0].real - block.omega[1].real), converged)
        stopped = tight & (block.n_first > 0)
        early_d = np.abs(block.omega_first - block.omega).max(axis=0)
        early = _max(early_d, stopped)
        n_bitequal = int((stopped & (block.omega_first == block.omega).all(axis=0)).sum())
        med = float(np.median(block.n_last[tight])) if tight.any() else math.nan
        lines.append(
            f"{cfl:5.2f}  {int(converged.sum()):3d}  {int((block.resid > 1e-12).sum()):3d}  "
            f"{int((block.resid > 1e-8).sum()):3d}  {int((block.resid > 1e-4).sum()):3d}  {med:6.1f}  "
            f"{d_his_c:.2e}  {d_his_a:.2e}  {d_build_c:.2e}  {d_build_a:.2e}  "
            f"{norm_c:.2e}  {norm_m:.2e}  {norm_c12:.2e}  {min_im:+.3e}  {at_alpha:.4f}  {n_grow:2d}  "
            f"{parity:.2e}  {parity_re:.2e}  {early:.2e}  {n_bitequal:2d}  "
            f"{block.seconds:7.1f}  {block.seconds_advection:7.1f}"
        )
    growing = [cfl for cfl in sorted(stability) if stability[cfl][2] > 0]
    first_growth = growing[0] if growing else None
    stable_before = [cfl for cfl in sorted(stability) if first_growth is None or cfl < first_growth]
    lines += [
        f"# max |d| against his table: converged rows {_or_na(his_converged)}, all rows {_or_na(his_all)}",
        f"# max |d| against our build: converged rows {_or_na(build_converged)}, all rows {_or_na(build_all)}",
        "# normalised against our build, CFL <= 0.5 (component, modulus): converged rows "
        f"{_or_na(normalised[0])}, {_or_na(normalised[1])}; resid < 1e-12 rows "
        f"{_or_na(normalised_tight[0])}, {_or_na(normalised_tight[1])}",
        f"# stability: last CFL without growth before the first growth {stable_before[-1] if stable_before else None}, "
        f"first CFL with growth {first_growth}",
        "# convergence: rows with resid > 1e-12 / 1e-8 / 1e-4: "
        + " / ".join(
            str(sum(int((b.resid > t).sum()) for b in blocks)) for t in (1e-12, 1e-8, 1e-4)
        )
        + f" of {len(blocks) * NUM_LEVELS_30}",
    ]
    return _Theta30Stats(
        lines=lines,
        his_converged=float(his_converged),
        his_all=float(his_all),
        build_converged=float(build_converged),
        build_all=float(build_all),
        build_normalised=(float(normalised[0]), float(normalised[1])),
        build_normalised_tight=(float(normalised_tight[0]), float(normalised_tight[1])),
        stability=stability,
    )


def _plot_theta30(blocks: list[_Theta30Block], scheme: int, tag: str) -> list[pathlib.Path]:
    """Figure 7 at theta = 30 (-Im omega at 695, non-converged rows grey), and next to theta = 0."""
    if plt is None:
        return []
    alphas = np.asarray(ALPHA_30)
    cfls = np.asarray([b.cfl for b in blocks])
    growth30 = np.stack(
        [np.where(b.resid <= CONVERGED_RESID_30, -b.omega[0].imag, np.nan) for b in blocks]
    )
    title30 = (
        f"ihadv_tracer {scheme}, theta = 30: growth rate at 695 (black: zero, grey: not converged)"
    )
    written = []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _growth_panel(fig=fig, ax=ax, alphas=alphas, cfls=cfls, growth=growth30, title=title30)
    path = OUTPUT_DIR / f"fig7_{_SCHEME_IDS[scheme]}_theta30_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    theta0_path = OUTPUT_DIR / f"{_SCHEME_IDS[scheme]}_theta0_{BACKEND_REFERENCE_TAG}.txt"
    if theta0_path.exists():
        theta0 = _read_fortran_table(theta0_path)
        cfls0 = np.asarray(CFL_FULL)
        growth0 = np.stack([-_fortran_block(theta0, cfl)[:, 3] for cfl in cfls0])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        _growth_panel(
            fig=fig,
            ax=axes[0],
            alphas=np.asarray(ALPHA),
            cfls=cfls0,
            growth=growth0,
            title=f"ihadv_tracer {scheme}, theta = 0: growth rate at 294 (black: zero)",
        )
        _growth_panel(fig=fig, ax=axes[1], alphas=alphas, cfls=cfls, growth=growth30, title=title30)
        path = OUTPUT_DIR / f"fig7_{_SCHEME_IDS[scheme]}_theta0_theta30_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _run_or_resume(
    *,
    setup: _Setup,
    scheme: int,
    cfls: tuple[float, ...],
    parts_dir: pathlib.Path,
    reuse: bool,
    deadline: float,
    iterations: int,
    label: str,
) -> tuple[dict[float, _Theta30Block], list[float], int]:
    """Every CFL's block, saved as soon as it is done; returns (blocks, CFLs left, computed).

    With ``reuse`` a saved CFL is loaded instead of recomputed. No CFL is started that
    would not end before ``deadline`` (epoch seconds) at the duration of the longest CFL
    so far (ICON4PY_DISPERSION_CFL_SECONDS, default 300, before the first).
    """
    first_estimate = float(os.environ.get("ICON4PY_DISPERSION_CFL_SECONDS", "300"))
    blocks: dict[float, _Theta30Block] = {}
    pending: list[float] = []
    longest = 0.0
    computed = 0
    for cfl in cfls:
        part = _part_path(parts_dir, scheme, cfl)
        if reuse and part.exists():
            blocks[cfl] = _load_part(part)
            assert blocks[cfl].iterations == iterations, part
            continue
        if pending or wall_time.time() + (longest or first_estimate) > deadline:
            pending.append(cfl)
            continue
        block = _chequerboard_iteration(setup, cfl, iterations)
        _save_part(part, block)
        blocks[cfl] = block
        computed += 1
        longest = max(longest, block.seconds)
        print(
            f"{label}: CFL {cfl:.2f} done in {block.seconds:.1f} s (Advection.run "
            f"{block.seconds_advection:.1f} s), rows resid > 1e-12: "
            f"{int((block.resid > 1e-12).sum())}",
            flush=True,
        )
    return blocks, pending, computed


def _env_cfls() -> tuple[float, ...] | None:
    value = os.environ.get("ICON4PY_DISPERSION_THETA30_CFLS", "").strip()
    return tuple(float(item) for item in value.split(",")) if value else None


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.skipif(not GRID_FILE.exists(), reason=f"Jocksch's grid file {GRID_FILE} not found")
@pytest.mark.parametrize("scheme", [2, 3], ids=_SCHEME_IDS.get)
@pytest.mark.parametrize("cfl_set", list(CFL_SETS_30))
def test_dispersion_relation_theta30(
    scheme: int,
    cfl_set: str,
    *,
    tmp_path: pathlib.Path,
    process_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend | None,
) -> None:
    """His chequerboard iteration at theta = 30 degrees (figure 7's stability limit).

    'paper': the six CFLs of figures 5 and 6, against his tables; 'stability': no growth at
    CFL 0.42, growth at 0.44 (converged rows), and against his tables; 'env': the CFLs of
    ICON4PY_DISPERSION_THETA30_CFLS for the job scripts (skipped when unset), which keep
    the CFLs already saved under theta30_parts/full_<backend>/ and stop starting new ones
    after ICON4PY_DISPERSION_DEADLINE (epoch seconds); with every CFL of the full list
    present they write the full table. ICON4PY_DISPERSION_REUSE_PARTS=1 lets 'paper' and
    'stability' reuse their saved CFLs too (to re-check the gates without the iteration).
    """
    backend_tag = data_alloc.backend_name(backend)
    stem = f"{_SCHEME_IDS[scheme]}_theta30"
    if cfl_set == "env":
        cfls = _env_cfls()
        if cfls is None:
            pytest.skip("ICON4PY_DISPERSION_THETA30_CFLS not set (job-script set)")
        name = "full" if sorted(cfls) == sorted(CFL_FULL) else "env"
        reuse = True
        deadline = float(os.environ.get("ICON4PY_DISPERSION_DEADLINE", "inf"))
        # a smoke-test override; his 1000 otherwise
        iterations = int(os.environ.get("ICON4PY_DISPERSION_THETA30_ITERATIONS", NUM_ITERATIONS_30))
        if iterations != NUM_ITERATIONS_30:
            name = f"{name}_it{iterations}"
        # every job of the chain saves into and resumes from the same directory
        parts_dir = PARTS_DIR / (
            f"full_{backend_tag}"
            if iterations == NUM_ITERATIONS_30
            else f"full_{backend_tag}_it{iterations}"
        )
    else:
        cfls = CFL_SETS_30[cfl_set]
        assert cfls is not None
        name = cfl_set
        reuse = os.environ.get("ICON4PY_DISPERSION_REUSE_PARTS", "0") not in ("", "0")
        deadline = math.inf
        iterations = NUM_ITERATIONS_30
        parts_dir = PARTS_DIR / f"{name}_{backend_tag}"
    tag = f"{name}_{backend_tag}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = wall_time.perf_counter()
    setup = _setup(scheme, backend, process_props, tmp_path, theta=30)
    setup_seconds = wall_time.perf_counter() - start
    print(f"\n{stem} {tag}: setup {setup_seconds:.1f} s", flush=True)

    blocks, pending, computed = _run_or_resume(
        setup=setup,
        scheme=scheme,
        cfls=cfls,
        parts_dir=parts_dir,
        reuse=reuse,
        deadline=deadline,
        iterations=iterations,
        label=f"{stem} {tag}",
    )
    if pending:
        pytest.skip(
            f"deadline reached: {len(blocks)} CFLs saved in {parts_dir}, {len(pending)} left "
            f"({', '.join(f'{c:.2f}' for c in pending)})"
        )
    ordered = [blocks[cfl] for cfl in cfls]
    for block in ordered:
        assert np.isfinite(block.omega).all(), f"CFL {block.cfl}: non-finite omega"

    reference_path = REFERENCE_TABLES_30[scheme]
    reference = _read_fortran_table(reference_path) if reference_path.exists() else None
    build_dir = FORTRAN_BUILD_DIRS_30[scheme]
    build = (
        _read_fortran_table(build_dir / "dispersion.txt")
        if (build_dir / "dispersion.txt").exists()
        else None
    )
    build_conv = _read_conv(build_dir / "conv.txt") if (build_dir / "conv.txt").exists() else None
    stats = _theta30_statistics(ordered, reference, build, build_conv)

    outputs = [(OUTPUT_DIR / f"{stem}_{tag}.txt", OUTPUT_DIR / f"{stem}_{tag}_conv.txt")]
    if backend_tag == "run_gtfn_cpu" and name in ("paper", "full"):
        suffix = "" if name == "paper" else "_full"
        outputs.append(
            (OUTPUT_DIR / f"{stem}{suffix}.txt", OUTPUT_DIR / f"{stem}{suffix}_conv.txt")
        )
    for table_path, conv_path in outputs:
        _write_theta30_table(table_path, ordered)
        _write_theta30_conv(conv_path, ordered, backend_tag)
    header = [
        f"# scheme {scheme}, theta = 30, backend {backend_tag}, cfl set {name} "
        f"({len(cfls)} CFL x {NUM_LEVELS_30} alpha, {iterations} iterations), "
        f"reference {reference_path}, our build {build_dir}",
        f"# setup {setup_seconds:.1f} s, total {wall_time.perf_counter() - start:.1f} s "
        f"(CFLs computed in this process: {computed}, reused from {parts_dir}: {len(cfls) - computed})",
    ]
    summary = OUTPUT_DIR / f"{stem}_{tag}_summary.txt"
    summary.write_text("\n".join(header + stats.lines) + "\n")
    figures = _plot_theta30(ordered, scheme, tag) if len(ordered) > 6 else []
    print("\n".join(header + stats.lines))
    print("figures: " + ", ".join(str(f) for f in figures))

    if cfl_set == "stability":
        min_im_042, _, n_grow_042, _ = stats.stability[0.42]
        min_im_044, _, n_grow_044, _ = stats.stability[0.44]
        assert n_grow_042 == 0, f"growth at CFL 0.42: min Im omega {min_im_042:.3e}"
        assert n_grow_044 > 0, f"no growth at CFL 0.44: min Im omega {min_im_044:.3e}"
        if reference is None:
            pytest.skip(
                f"growth limits asserted; Fortran reference table {reference_path} not available"
            )
        assert stats.his_converged <= FORTRAN_TOLERANCE_30[scheme], stats.his_converged
    if cfl_set == "paper":
        if reference is None:
            pytest.skip(f"Fortran reference table {reference_path} not available")
        assert stats.his_converged <= FORTRAN_TOLERANCE_30[scheme], stats.his_converged

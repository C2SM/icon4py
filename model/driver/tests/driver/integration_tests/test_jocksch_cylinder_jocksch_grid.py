# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Jocksch's moving-cylinder experiment on his own grid: the rows of the paper's Table 2.

The experiment of test_jocksch_cylinder.py (which runs it on the generated torus) on the
grid the numbers of the paper (Jocksch et al., PPAM 2026) come from: his
torus_grid_r4_c200_elen100.nc, 880 cells / 1320 edges / 440 vertices, 5 km edges, centred
coordinates (cell centres x in [-52.5, 47.5] km, y in +-46.2 km) and every edge normal with
n_x >= 0. The cylinder sits at the origin as in his runs (176 cells inside), the wind is
+x, so mass_flx_me >= 0 on every edge, which is the condition his cell-local positive-
definite limiter needs (the generated grid has 440 edges with vn < 0, where his limiter
zeroes the flux). This module therefore adds the rows the generated grid cannot check: the
weight sets, the hybrid scheme and Jocksch's limiter; the 2/3/102 rows are regression rows
against the Fortran runs on this grid.

Reference data: weno_data/reference/jocksch_grid/<case>/error.txt with case names
ihadv<scheme>_hlim<limiter>[_dj1|_ones] (his printed '#' error, the neighbour-pair sum of
test_jocksch_cylinder.py); the results table is in icon-ajocksch/CAPTURE_NOTES.md.
"""

from __future__ import annotations

import math
import pathlib
from typing import TYPE_CHECKING, Final

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection, weno_least_squares
from icon4py.model.common.decomposition import definitions as decomp_defs

from .. import utils as test_utils
from ..fixtures import *  # noqa: F403
from .test_jocksch_cylinder import PAPER_TRUNCATION


if TYPE_CHECKING:
    # what pytest.param returns; pytest 9 does not re-export it, so the private import
    # stays a type-checking one (the annotations are strings, see __future__ above)
    from _pytest.mark import ParameterSet


#: Andreas Jocksch's own torus (a copy of his dispersion_relation/icon/grids/
#: torus_grid_r4_c200_elen100.nc; the name is misleading): 20 x 22, 5 km edges, centred
GRID_FILE: Final = pathlib.Path(
    "/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim/weno_data/grids/"
    "jocksch_torus_grid_r4_c200_elen100.nc"
)
#: the Fortran reference runs on this grid, <case>/error.txt with his printed '#' error
REFERENCE_DIR: Final = GRID_FILE.parents[1] / "reference" / "jocksch_grid"
#: his cylinder is centred at the origin (the cell nearest to it is at (0, -1443.4) m)
CYLINDER_CENTER: Final[tuple[float, float]] = (0.0, 0.0)

_HADV = tracer_advection.HorizontalAdvectionType
_HLIM = tracer_advection.HorizontalAdvectionLimiter
_WEIGHTS = weno_least_squares.WenoLinearWeights
_Case = tuple[_HADV, _HLIM, _WEIGHTS]

#: Table 2 of the paper by (scheme, limiter, weight set): sqrt(pair sum / 3) printed with
#: three decimals (truncated in some entries, rounded in others, see CAPTURE_NOTES.md; the
#: gate is the union of both readings). The weight set only matters for 103 and 132.
#: "WENO d_j = 1" is the all-ones set (UNITY); the hybrid column is reproduced by the UNITY
#: set (3.3106), not by the optimised one (3.3124). The paper's 3.309 for "WENO d_j = 1 +
#: limiter" is not reproduced by the Fortran either (3.3084, PAPER_TABLE_2_NOT_REPRODUCED).
PAPER_TABLE_2: Final[dict[_Case, float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 4.023,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.402,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.859,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.058,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 3.310,
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 3.310,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 3.842,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 3.308,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 3.003,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 3.371,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 3.361,
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 3.778,
}
#: paper entries the Fortran reference run itself does not hit (printed, not gated)
PAPER_TABLE_2_NOT_REPRODUCED: Final[dict[_Case, float]] = {
    # ihadv103_hlim4_ones prints 32.83643759358356, i.e. 3.3084; the paper says 3.309
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.UNITY): 3.309,
    # ihadv132_hlim0 (opt set) prints 32.91546685721701, i.e. 3.3124; the paper says 3.310
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.310,
}

#: the '#' error the Fortran reference runs print on this grid,
#: weno_data/reference/jocksch_grid/<case>/error.txt (_ones = UNITY, _dj1 = HAND_TUNED)
FORTRAN_ERROR_SUM: Final[dict[_Case, float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 48.54601406791249,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 34.73851779887965,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 44.68350118207169,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 28.06422509932473,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 32.88393941867035,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.HAND_TUNED): 32.24012112150874,
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 32.88020252438776,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.NO_LIMITER,
        _WEIGHTS.OPTIMIZED,
    ): 32.91546685721701,
    (
        _HADV.LINEAR_2ND_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 44.27477978369598,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 32.83643759358356,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 32.83291364341792,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 27.04602215516110,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 34.09527697903994,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 33.90049401088558,
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 42.82532613695046,
}
#: relative agreement with FORTRAN_ERROR_SUM, gated at about three times the value
#: measured on gtfn_cpu (in the comments) to leave room for the other backends. dt is
#: 1000 s here against 999.99999995 s there (the 1e-11 to 1e-10 of the non-WENO rows); the
#: WENO smoothness indicator and the hybrid's fit residual are single precision in the
#: Fortran and working precision here (the 1e-9 of the 103/132 rows, see
#: test_jocksch_reference.py). The test prints the difference per case.
FORTRAN_ERROR_RTOL: Final[dict[_Case, float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 1e-10,  # 3.0e-11
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3e-10,  # 7.7e-11
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 1e-9,  # 3.3e-10
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 2e-8,  # 4.0e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 6e-9,  # 1.8e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.HAND_TUNED): 8e-9,  # 2.4e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 8e-9,  # 2.3e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3e-9,  # 8.1e-10
    (
        _HADV.LINEAR_2ND_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 1e-9,  # 3.3e-10
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 6e-9,  # 2.0e-9
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 8e-9,  # 2.2e-9
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 2e-8,  # 3.9e-9
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 1e-10,  # 3.2e-11
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 2e-10,  # 4.8e-11
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 6e-10,  # 1.7e-10
}


def _param(hadv: _HADV, hlim: _HLIM, weights: _WEIGHTS = _WEIGHTS.OPTIMIZED) -> ParameterSet:
    weight_tag = (
        ""
        if hadv not in (_HADV.QUADRATIC_3RD_ORDER_WENO, _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID)
        else f"-{weights.name.lower()}"
    )
    return pytest.param(hadv, hlim, weights, id=f"ihadv{hadv.value}-hlim{hlim.value}{weight_tag}")


CASES: Final[list[ParameterSet]] = [
    # regression rows: the schemes the generated grid already gates
    _param(_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER),
    _param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER),
    _param(_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER),
    _param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC),
    _param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE),
    _param(_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC),
    # the weight sets of the quadratic WENO scheme
    _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED),
    _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY),
    _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.HAND_TUNED),
    # the hybrid scheme
    _param(_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY),
    _param(_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED),
    # Jocksch's cell-local positive-definite limiter
    _param(_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE),
    _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.UNITY),
    _param(
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ),
    _param(
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ),
]


def _reference_case_name(case: _Case) -> str:
    """The case directory of the Fortran run: ihadv<scheme>_hlim<limiter>[_dj1|_ones].

    Jocksch's cell-local limiter is his itype_hlimit=4 inside the WENO schemes (hlim4, the
    positive-definite limiter's number); the weight-set suffix exists for 103 and 132 only.
    """
    hadv, hlim, weights = case
    itype_hlimit = {
        _HLIM.NO_LIMITER: 0,
        _HLIM.MONOTONIC: 3,
        _HLIM.POSITIVE_DEFINITE: 4,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE: 4,
    }[hlim]
    suffix = ""
    if hadv in (_HADV.QUADRATIC_3RD_ORDER_WENO, _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID):
        suffix = {_WEIGHTS.OPTIMIZED: "", _WEIGHTS.UNITY: "_ones", _WEIGHTS.HAND_TUNED: "_dj1"}[
            weights
        ]
    return f"ihadv{hadv.value}_hlim{itype_hlimit}{suffix}"


@pytest.mark.parametrize(
    "horizontal_advection_type, horizontal_advection_limiter, weno_linear_weights", CASES
)
def test_fortran_error_sum_matches_reference_file(
    horizontal_advection_type: _HADV,
    horizontal_advection_limiter: _HLIM,
    weno_linear_weights: _WEIGHTS,
) -> None:
    """The typed-in FORTRAN_ERROR_SUM is the number in the reference run's error.txt."""
    case: _Case = (horizontal_advection_type, horizontal_advection_limiter, weno_linear_weights)
    error_file = REFERENCE_DIR / _reference_case_name(case) / "error.txt"
    if not error_file.exists():
        pytest.skip(f"Fortran reference output {error_file} not available")
    # the file holds one line, " #    <pair sum>"
    printed = float(error_file.read_text().strip().lstrip("#").strip())
    assert printed == FORTRAN_ERROR_SUM[case]


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.skipif(not GRID_FILE.exists(), reason=f"Jocksch's grid file {GRID_FILE} not found")
@pytest.mark.parametrize(
    "horizontal_advection_type, horizontal_advection_limiter, weno_linear_weights", CASES
)
def test_jocksch_cylinder_one_period_on_jocksch_grid(
    horizontal_advection_type: _HADV,
    horizontal_advection_limiter: _HLIM,
    weno_linear_weights: _WEIGHTS,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    case: _Case = (horizontal_advection_type, horizontal_advection_limiter, weno_linear_weights)
    run = test_utils.run_cylinder_one_period(
        grid_file=GRID_FILE,
        cylinder_center=CYLINDER_CENTER,
        tracer_advection={
            "horizontal_advection_type": horizontal_advection_type,
            "horizontal_advection_limiter": horizontal_advection_limiter,
            "weno_linear_weights": weno_linear_weights,
        },
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
    )
    # the cylinder is a full disc of 176 cells at the origin, and his +x wind gives a
    # non-negative normal mass flux on every edge of this grid (n_x >= 0 everywhere), the
    # condition his cell-local limiter needs (mass_flx_me = u * n_x with u > 0)
    assert int(run.cylinder.sum()) == 176
    assert (run.edge_normal_x >= 0.0).all(), "his cell-local limiter needs mass_flx_me >= 0"

    jocksch_measure = run.jocksch_measure
    fortran_value = FORTRAN_ERROR_SUM[case]
    relative_difference = abs(jocksch_measure - fortran_value) / fortran_value
    paper_value = PAPER_TABLE_2.get(case)
    paper_not_reproduced = PAPER_TABLE_2_NOT_REPRODUCED.get(case)
    print(
        f"\n{horizontal_advection_type.name} ({horizontal_advection_type.value}) + "
        f"{horizontal_advection_limiter.name} ({horizontal_advection_limiter.value}), "
        f"weights {weno_linear_weights.name}: {test_utils.CYLINDER_N_TIME_STEPS} steps of "
        f"dt = {run.dtime_seconds} s, {run.num_levels} level(s), "
        f"wall time {run.elapsed_wall_time:.1f} s\n"
        f"  Jocksch measure (pairs within an edge length)  = {jocksch_measure:.6f}"
        f"  sqrt(/3) = {math.sqrt(jocksch_measure / 3.0):.6f}\n"
        f"  all neighbour pairs (= 3 sum e^2)              = {run.all_pairs_measure:.6f}\n"
        f"  sum e^2                                        = {run.sum_squared_error:.6f}\n"
        f"  overshoot (max q - 1) final / run              = {run.overshoot_final:.6e} / "
        f"{run.overshoot_run:.6e}\n"
        f"  undershoot (-min q) final / run                = {run.undershoot_final:.6e} / "
        f"{run.undershoot_run:.6e}\n"
        f"  relative mass change                           = {run.relative_mass_change:.6e}\n"
        f"  Fortran reference (his grid)                   = {fortran_value}"
        f"  sqrt(/3) = {math.sqrt(fortran_value / 3.0):.6f}\n"
        f"  relative difference to the Fortran pair sum    = {relative_difference:.3e}\n"
        f"  paper Table 2                                  = {paper_value}"
        + (f" (paper prints {paper_not_reproduced}, not gated)" if paper_not_reproduced else "")
    )

    # the paper prints three decimals, truncated or rounded: gate on the union of both
    if paper_value is not None:
        root = math.sqrt(jocksch_measure / 3.0)
        assert paper_value - 0.5 * PAPER_TRUNCATION <= root < paper_value + PAPER_TRUNCATION, (
            f"sqrt(pair sum / 3) = {root:.6f} does not print as the paper's {paper_value}"
        )
    assert relative_difference <= FORTRAN_ERROR_RTOL[case]

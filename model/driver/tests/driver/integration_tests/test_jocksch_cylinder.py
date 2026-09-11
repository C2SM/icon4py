# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Jocksch's moving-cylinder experiment for the FFSL-WENO schemes on the shared torus.

A tracer cylinder (radius 25 km) is carried once around a periodic 20 x 22 torus with
5 km edges by a uniform 1 m/s wind: with dt = CFL * edge_length = 1000 s, 100 steps are
exactly one period in x, so the exact solution is the initial field and the error is
q(100) - q(0). The grid file is the one the Fortran reference run uses, so the numbers
printed here can be set next to Table 2 of the paper and to the Fortran runs; both are
asserted ('assert_table_2_gates'), and the error table is printed (run with '-s'). The run
itself and the experiment's constants are shared with test_jocksch_cylinder_jocksch_grid.py
('utils.run_cylinder_one_period').

The error measures follow the live block in Jocksch's 'mo_nh_stepping.f90': his measure
sums (e_i^2 + e_j^2) over the unordered neighbour-cell pairs (i, j) whose raw (non-
periodic) centre distance is below the edge length, which drops the pairs across the
periodic boundary; with those pairs it would be exactly 3 * sum(e^2) on a torus.
"""

import math
import pathlib
from typing import Final

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common.decomposition import definitions as decomp_defs

from .. import utils as test_utils
from ..fixtures import *  # noqa: F403


#: the planar torus shared with the Fortran reference run: 20 x 22, 5 km edges,
#: 880 cells / 1320 edges / 440 vertices, 100 km x 95.26 km
GRID_FILE: Final = pathlib.Path(
    "/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim/weno_data/grids/"
    "torus_20x22_res5000m.nc"
)

#: cylinder centre in the grid file's coordinates, None is the domain centre
CYLINDER_CENTER: Final[tuple[float | None, float | None]] = (None, None)

_HADV = tracer_advection.HorizontalAdvectionType
_HLIM = tracer_advection.HorizontalAdvectionLimiter

#: Table 2 of the paper, by (scheme, limiter): sqrt(Jocksch's pair sum / 3) truncated to
#: three decimals. The Fortran reference runs settle the normalisation (FORTRAN_ERROR_SUM):
#: the pair sum is 3 sum e^2 without the pairs across the periodic seam, which matters for
#: the linear scheme, whose error reaches the seam (sqrt(sum e^2) = 4.0240 there). The
#: limiter rows of Table 2 for the WENO schemes use Jocksch's own cell-local limiter, which
#: icon4py does not have, so only the rows of schemes 2 and 3 are gated.
PAPER_TABLE_2: Final[dict[tuple[_HADV, _HLIM], float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER): 4.023,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER): 3.859,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER): 3.402,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER): 3.058,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC): 3.371,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE): 3.361,
    # the paper's "linear lsq, with limiter" is the monotonic one: the Fortran runs give
    # sqrt(42.82697 / 3) = 3.7783 with hflx_limiter_mo and 3.792 with hflx_limiter_pd
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC): 3.778,
}
#: the paper prints sqrt(sum e^2) truncated to three decimals
PAPER_TRUNCATION: Final = 1e-3

#: the '#' error the Fortran reference runs print, weno_data/reference/<case>_centred/error.txt:
#: Jocksch's neighbour-pair sum over the pairs within an edge length (the ones across the
#: periodic seam dropped), 100 steps of the same experiment on the coordinate-shifted copy
#: of this grid file, where his cylinder at the origin is the domain-centred one used here.
#: Both runs skip the same pairs, since the seam is the same set of mesh edges.
FORTRAN_ERROR_SUM: Final[dict[tuple[_HADV, _HLIM], float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER): 48.56183199462232,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER): 44.68376363352618,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER): 34.73852533016048,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER): 28.06422915857846,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC): 34.09527697903955,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE): 33.90049401088627,
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC): 42.82697018834337,
    (_HADV.LINEAR_2ND_ORDER, _HLIM.POSITIVE_DEFINITE): 43.14422010497015,
}
#: relative agreement with FORTRAN_ERROR_SUM, gated at the value measured on gtfn_cpu
#: (in brackets) with a factor of three for the other backends; the differences between
#: the two experiments are dt = 1000 s here against 999.99999995 s there (5e-11), and for
#: 103 the Fortran's single-precision WENO smoothness indicator (see
#: test_jocksch_reference.py). The test prints the relative difference per case.
FORTRAN_ERROR_RTOL: Final[dict[tuple[_HADV, _HLIM], float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER): 1e-10,  # 2.9e-11
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER): 1e-9,  # 3.3e-10
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER): 3e-10,  # 7.7e-11
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER): 1e-8,  # 3.5e-9
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC): 1e-10,  # 3.2e-11
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE): 2e-10,  # 4.8e-11
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC): 6e-10,  # 1.7e-10
    (_HADV.LINEAR_2ND_ORDER, _HLIM.POSITIVE_DEFINITE): 2e-10,  # 4.6e-11
}


def assert_table_2_gates(
    *,
    horizontal_advection_type: _HADV,
    horizontal_advection_limiter: _HLIM,
    jocksch_measure: float,
) -> None:
    """Gate Jocksch's error measure against the paper's Table 2 and the Fortran runs.

    The paper value is sqrt(pair sum / 3) truncated to three decimals, so the gate is the
    truncation interval [paper, paper + 0.001). The Fortran pair sum is gated at the
    case's FORTRAN_ERROR_RTOL.
    """
    key = (horizontal_advection_type, horizontal_advection_limiter)
    if (paper_value := PAPER_TABLE_2.get(key)) is not None:
        root = math.sqrt(jocksch_measure / 3.0)
        assert paper_value <= root < paper_value + PAPER_TRUNCATION, (
            f"sqrt(pair sum / 3) = {root:.6f} does not truncate to the paper's {paper_value}"
        )
    if (fortran_value := FORTRAN_ERROR_SUM.get(key)) is not None:
        relative_difference = abs(jocksch_measure - fortran_value) / fortran_value
        print(f"  relative difference to the Fortran pair sum    = {relative_difference:.3e}")
        assert relative_difference <= FORTRAN_ERROR_RTOL[key]


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.skipif(not GRID_FILE.exists(), reason=f"shared grid file {GRID_FILE} not found")
@pytest.mark.parametrize(
    "horizontal_advection_type, horizontal_advection_limiter",
    [
        pytest.param(_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, id="miura"),
        pytest.param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, id="miura3"),
        pytest.param(_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, id="miura_weno"),
        pytest.param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, id="miura3_weno"),
        pytest.param(_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, id="miura-monotonic"),
        pytest.param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, id="miura3-monotonic"),
        pytest.param(_HADV.LINEAR_2ND_ORDER, _HLIM.POSITIVE_DEFINITE, id="miura-positive_definite"),
        pytest.param(
            _HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, id="miura3-positive_definite"
        ),
    ],
)
def test_jocksch_cylinder_one_period(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
    horizontal_advection_limiter: tracer_advection.HorizontalAdvectionLimiter,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    run = test_utils.run_cylinder_one_period(
        grid_file=GRID_FILE,
        cylinder_center=CYLINDER_CENTER,
        tracer_advection={
            "horizontal_advection_type": horizontal_advection_type,
            "horizontal_advection_limiter": horizontal_advection_limiter,
        },
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
    )

    paper_value = PAPER_TABLE_2.get((horizontal_advection_type, horizontal_advection_limiter))
    print(
        f"\n{horizontal_advection_type.name} ({horizontal_advection_type.value}) + "
        f"{horizontal_advection_limiter.name} ({horizontal_advection_limiter.value}): "
        f"{test_utils.CYLINDER_N_TIME_STEPS} steps of dt = {run.dtime_seconds} s, "
        f"{run.num_levels} level(s), wall time {run.elapsed_wall_time:.1f} s\n"
        f"  Jocksch measure (pairs within an edge length)  = {run.jocksch_measure:.6f}"
        f"  sqrt = {math.sqrt(run.jocksch_measure):.6f}\n"
        f"  all neighbour pairs (= 3 sum e^2)              = {run.all_pairs_measure:.6f}"
        f"  sqrt = {math.sqrt(run.all_pairs_measure):.6f}\n"
        f"  sum e^2                                        = {run.sum_squared_error:.6f}"
        f"  sqrt = {math.sqrt(run.sum_squared_error):.6f}\n"
        f"  overshoot (max q - 1) final / run              = {run.overshoot_final:.6e} / "
        f"{run.overshoot_run:.6e}\n"
        f"  undershoot (-min q) final / run                = {run.undershoot_final:.6e} / "
        f"{run.undershoot_run:.6e}\n"
        f"  relative mass change                           = {run.relative_mass_change:.6e}\n"
        f"  paper Table 2                                  = {paper_value}\n"
        f"  Fortran reference (centred run)                = "
        f"{FORTRAN_ERROR_SUM.get((horizontal_advection_type, horizontal_advection_limiter))}"
    )
    assert_table_2_gates(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        jocksch_measure=run.jocksch_measure,
    )

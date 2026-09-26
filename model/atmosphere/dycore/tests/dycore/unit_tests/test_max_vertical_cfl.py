# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
`max_vertical_cfl` (ICON's `max_vcfl_dyn`) is the maximum of the vertical CFL *magnitude*, as in
ICON's `maxvcfl = MAX(maxvcfl, ABS(vcfl))` (`mo_velocity_advection.f90`).

The serialized data has no CFL clipping, so the datatests cannot tell a signed from an absolute
reduction. These tests run the reductions of `run_predictor_step` and `run_corrector_step` with
the GT4Py programs replaced by stubs. The stub of the vertical momentum program computes
`vertical_cfl` with the granule's own clipping field operator, from a prescribed contravariant
vertical wind.
"""

from typing import Any
from unittest import mock

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore import velocity_advection as advection
from icon4py.model.atmosphere.dycore.stencils import (
    compute_advection_in_vertical_momentum_equation as vertical_momentum,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


DTIME = 2.0
# With DTIME = 2 s and the granule's cfl_w_limit = 0.65, a point is clipped (and so counts in
# the CFL maximum) where |w_con| > 0.65 / DTIME * DDQZ_Z_HALF = 6.5 m/s.
DDQZ_Z_HALF = 20.0


def _cell_k_field(values: Any, dtype: Any = ta.vpfloat) -> gtx.Field:
    return gtx.as_field(
        (dims.CellDim, dims.KDim),
        np.asarray(values, dtype=dtype),  # type: ignore[arg-type]
    )


def _max_vertical_cfl(
    step: str,
    w_con: list[list[float]],
    previous: float = 0.0,
    horizontal_start: int = 0,
    horizontal_end: int | None = None,
) -> float:
    shape = np.shape(w_con)

    def compute_vertical_cfl(
        *, vertical_cfl: gtx.Field, cfl_w_limit: float, dtime: float, **kwargs: Any
    ) -> None:
        vertical_momentum._compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
            _cell_k_field(np.full(shape, DDQZ_Z_HALF)),
            _cell_k_field(w_con),
            ta.vpfloat(cfl_w_limit),
            ta.wpfloat(dtime),
            out=(
                _cell_k_field(np.zeros(shape)),
                _cell_k_field(np.zeros(shape), bool),
                vertical_cfl,
            ),
            offset_provider={},
        )

    # No __init__: only the attributes that the two run methods read are set.
    granule: Any = object.__new__(advection.VelocityAdvection)
    granule._cfl_w_limit = 0.65
    granule._scalfac_exdiff = 0.05
    granule._vertical_cfl = _cell_k_field(np.zeros(shape))
    # only passed on to the stubbed programs
    granule._contravariant_corrected_w_at_cells_on_model_levels = None
    granule._horizontal_advection_of_w_at_edges_on_half_levels = None
    granule._start_cell_lateral_boundary_level_4 = horizontal_start
    granule._end_cell_halo = shape[0] if horizontal_end is None else horizontal_end
    granule._compute_diagnostics_from_normal_wind = mock.Mock()
    granule._compute_advection_in_predictor_vertical_momentum = compute_vertical_cfl
    granule._compute_advection_in_corrector_vertical_momentum = compute_vertical_cfl
    granule._compute_advection_in_horizontal_momentum = mock.Mock()

    diagnostic_state = mock.Mock()
    diagnostic_state.max_vertical_cfl = data_alloc.scalar_like_array(previous)
    common_args = {
        "diagnostic_state": diagnostic_state,
        "prognostic_state": mock.Mock(),
        "horizontal_kinetic_energy_at_edges_on_model_levels": None,
        "tangential_wind_on_half_levels": None,
        "dtime": DTIME,
        "cell_areas": None,
    }
    if step == "predictor":
        granule.run_predictor_step(
            skip_compute_predictor_vertical_advection=False,
            contravariant_correction_at_edges_on_model_levels=None,
            **common_args,
        )
    else:
        granule.run_corrector_step(**common_args)
    return float(diagnostic_state.max_vertical_cfl)


steps = pytest.mark.parametrize("step", ["predictor", "corrector"])


@steps
@pytest.mark.parametrize(
    "w_con, expected",
    [
        # a clipped downdraft of CFL -2.0 is the largest; the signed maximum would give 0.8
        ([[-20.0, 8.0], [3.0, 0.0]], 2.0),
        # clipped downdrafts only; the signed maximum would give 0.0
        ([[-20.0, -8.0], [-3.0, 0.0]], 2.0),
        ([[10.0, -3.0], [8.0, 0.0]], 1.0),
        # nothing is clipped: |w_con| <= 6.5 m/s everywhere
        ([[-6.0, 6.0], [-3.0, 0.0]], 0.0),
    ],
    ids=["extreme_is_a_downdraft", "only_downdrafts", "only_updrafts", "no_clipping"],
)
def test_reduces_the_magnitude(step: str, w_con: list[list[float]], expected: float) -> None:
    assert _max_vertical_cfl(step, w_con) == expected


@steps
def test_keeps_the_running_maximum(step: str) -> None:
    assert _max_vertical_cfl(step, [[-20.0, 0.0], [8.0, 0.0]], previous=2.5) == 2.5


@steps
def test_ignores_cells_outside_the_horizontal_range(step: str) -> None:
    w_con = [[-40.0, 0.0], [-20.0, 0.0], [-60.0, 0.0]]
    assert _max_vertical_cfl(step, w_con, horizontal_start=1, horizontal_end=2) == 2.0

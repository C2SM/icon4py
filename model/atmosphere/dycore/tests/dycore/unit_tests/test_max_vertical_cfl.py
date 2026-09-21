# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import typing

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore import solve_nonhydro
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import nonhydro_states
from icon4py.model.common.utils import data_allocation as data_alloc


@dataclasses.dataclass
class _DiagnosticState:
    """Stands in for `DiagnosticStateNonHydro`, of which only this field is read and written."""

    max_vertical_cfl: data_alloc.ScalarLikeArray[ta.wpfloat]  # type: ignore[type-var]


def _update(
    values: list[list[float]], previous: float = 0.0, horizontal_end: int | None = None
) -> float:
    vertical_cfl = gtx.as_field(
        (dims.CellDim, dims.KHalfDim),
        np.asarray(values, dtype=ta.wpfloat),  # type: ignore[arg-type]
    )
    state = _DiagnosticState(max_vertical_cfl=data_alloc.scalar_like_array(previous))
    solve_nonhydro._update_max_vertical_cfl(
        typing.cast(nonhydro_states.DiagnosticStateNonHydro, state),
        vertical_cfl,
        gtx.int32(0),
        gtx.int32(len(values) if horizontal_end is None else horizontal_end),
    )
    return float(state.max_vertical_cfl)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[-1.2, -0.3], [-0.9, -0.1]], 1.2),
        ([[-1.2, 0.4], [0.5, 0.0]], 1.2),
        ([[0.7, 0.4], [0.5, -0.2]], 0.7),
    ],
    ids=["only_downdrafts", "extreme_is_a_downdraft", "only_updrafts"],
)
def test_reduces_the_magnitude(values: list[list[float]], expected: float) -> None:
    assert _update(values) == expected


def test_keeps_the_running_maximum() -> None:
    assert _update([[-1.2, 0.0], [0.3, 0.0]], previous=2.5) == 2.5


def test_ignores_cells_outside_the_horizontal_range() -> None:
    assert _update([[0.3, 0.0], [-1.2, 0.0]], horizontal_end=1) == 0.3

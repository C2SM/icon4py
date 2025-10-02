# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.dycore import dycore_utils
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils

from ..fixtures import backend


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

# TODO(): apply StencilTest structure to this test


def fourth_order_divdamp_scaling_coeff_for_order_24_numpy(
    a: np.ndarray, factor: float, mean_cell_area: float
) -> np.ndarray:
    a = np.maximum(0.0, a - 0.25 * factor)
    return -a * mean_cell_area**2


def calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary_numpy(
    coeff: float, field: np.ndarray
) -> np.ndarray:
    return 0.75 / (coeff + constants.DBL_EPS) * np.abs(field)


def test_calculate_fourth_order_divdamp_scaling_coeff_order_24(
    backend: gtx_typing.Backend,
) -> None:
    second_order_divdamp_factor = 3.0
    divdamp_order = 24
    mean_cell_area = 1000.0
    grid = simple_grid.simple_grid(backend=backend)
    interpolated_fourth_order_divdamp_factor = data_alloc.random_field(
        grid, dims.KDim, backend=backend
    )
    out = data_alloc.random_field(grid, dims.KDim, backend=backend)

    dycore_utils._calculate_fourth_order_divdamp_scaling_coeff.with_backend(backend)(
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        second_order_divdamp_factor=second_order_divdamp_factor,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )

    ref = fourth_order_divdamp_scaling_coeff_for_order_24_numpy(
        interpolated_fourth_order_divdamp_factor.asnumpy(),
        second_order_divdamp_factor,
        mean_cell_area,
    )
    assert test_utils.dallclose(ref, out.asnumpy())


def test_calculate_fourth_order_divdamp_scaling_coeff_any_order(
    backend: gtx_typing.Backend,
) -> None:
    second_order_divdamp_factor = 4.2
    divdamp_order = 3
    mean_cell_area = 1000.0
    grid = simple_grid.simple_grid(backend=backend)
    interpolated_fourth_order_divdamp_factor = data_alloc.random_field(
        grid, dims.KDim, backend=backend
    )
    out = data_alloc.random_field(grid, dims.KDim, backend=backend)

    dycore_utils._calculate_fourth_order_divdamp_scaling_coeff.with_backend(backend)(
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        second_order_divdamp_factor=second_order_divdamp_factor,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )
    enhanced_factor = -interpolated_fourth_order_divdamp_factor.asnumpy() * mean_cell_area**2
    assert test_utils.dallclose(enhanced_factor, out.asnumpy())


def test_calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary(
    backend: gtx_typing.Backend,
) -> None:
    grid = simple_grid.simple_grid(backend=backend)
    fourth_order_divdamp_scaling_coeff = data_alloc.random_field(grid, dims.KDim, backend=backend)
    out = data_alloc.zero_field(grid, dims.KDim, backend=backend)
    coeff = 0.3
    dycore_utils._calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary.with_backend(
        backend
    )(fourth_order_divdamp_scaling_coeff, coeff, constants.DBL_EPS, out=out, offset_provider={})
    assert test_utils.dallclose(
        out.asnumpy(),
        calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary_numpy(
            coeff, fourth_order_divdamp_scaling_coeff.asnumpy()
        ),
    )


def test_calculate_divdamp_fields(backend: gtx_typing.Backend) -> None:
    grid = simple_grid.simple_grid(backend=backend)
    divdamp_field = data_alloc.random_field(grid, dims.KDim, backend=backend)
    fourth_order_divdamp_scaling_coeff = data_alloc.zero_field(grid, dims.KDim, backend=backend)
    reduced_fourth_order_divdamp_coeff_at_nest_boundary = data_alloc.zero_field(
        grid, dims.KDim, backend=backend
    )
    divdamp_order = gtx.int32(24)
    mean_cell_area = 1000.0
    second_order_divdamp_factor = 0.7
    max_nudging_coefficient = 0.3

    scaled_ref = fourth_order_divdamp_scaling_coeff_for_order_24_numpy(
        divdamp_field.asnumpy(), second_order_divdamp_factor, mean_cell_area
    )

    reduced_fourth_order_divdamp_coeff_at_nest_boundary_ref = (
        calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary_numpy(
            max_nudging_coefficient, scaled_ref
        )
    )

    dycore_utils._calculate_divdamp_fields.with_backend(backend)(
        divdamp_field,
        divdamp_order,
        mean_cell_area,
        second_order_divdamp_factor,
        max_nudging_coefficient,
        constants.WP_EPS,
        out=(
            fourth_order_divdamp_scaling_coeff,
            reduced_fourth_order_divdamp_coeff_at_nest_boundary,
        ),
        offset_provider={},
    )
    test_utils.dallclose(fourth_order_divdamp_scaling_coeff.asnumpy(), scaled_ref)
    test_utils.dallclose(
        reduced_fourth_order_divdamp_coeff_at_nest_boundary.asnumpy(),
        reduced_fourth_order_divdamp_coeff_at_nest_boundary_ref,
    )

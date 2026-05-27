# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common import type_alias as ta
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.interpolation import rbf_interpolation as rbf


def _compute_edge_coefficients(dtype: type[np.floating]) -> np.ndarray:
    return rbf.compute_rbf_interpolation_coeffs_edge(
        edge_lat=np.asarray([0.0], dtype=dtype),
        edge_lon=np.asarray([0.0], dtype=dtype),
        edge_center_x=np.asarray([0.0], dtype=dtype),
        edge_center_y=np.asarray([0.0], dtype=dtype),
        edge_center_z=np.asarray([0.0], dtype=dtype),
        edge_normal_x=np.asarray([1.0], dtype=dtype),
        edge_normal_y=np.asarray([0.0], dtype=dtype),
        edge_normal_z=np.asarray([0.0], dtype=dtype),
        edge_dual_normal_u=np.asarray([1.0], dtype=dtype),
        edge_dual_normal_v=np.asarray([0.0], dtype=dtype),
        rbf_offset=np.asarray([[0]], dtype=np.int32),
        rbf_kernel=rbf.InterpolationKernel.INVERSE_MULTIQUADRATIC,
        geometry_type=icon_grid.GeometryType.TORUS,
        scale_factor=dtype(1.0),
        horizontal_start=0,
        horizontal_end=1,
        domain_length=dtype(1.0),
        domain_height=dtype(1.0),
    )


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [("double", np.float64), ("single", np.float32)],
)
def test_compute_rbf_interpolation_coeffs_returns_wpfloat(
    precision: str, expected_dtype: type[np.floating]
) -> None:
    previous_precision = ta.precision
    try:
        ta.set_precision(precision)
        coeff = _compute_edge_coefficients(np.float32)
        assert coeff.dtype == expected_dtype
    finally:
        ta.set_precision(previous_precision)


def test_compute_rbf_interpolation_coeffs_solves_in_float64_under_single_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_precision = ta.precision
    solve_argument_dtypes: list[tuple[np.dtype, np.dtype]] = []
    array_ns = rbf.data_alloc.array_namespace(np.asarray([0.0], dtype=np.float32))
    original_solve = array_ns.linalg.solve

    def recording_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        solve_argument_dtypes.append((matrix.dtype, rhs.dtype))
        return original_solve(matrix, rhs)

    try:
        ta.set_precision("single")
        monkeypatch.setattr(array_ns.linalg, "solve", recording_solve)
        _compute_edge_coefficients(np.float32)
    finally:
        ta.set_precision(previous_precision)

    assert solve_argument_dtypes
    assert all(mat_dtype == np.float64 and rhs_dtype == np.float64 for mat_dtype, rhs_dtype in solve_argument_dtypes)

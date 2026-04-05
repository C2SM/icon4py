# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests comparing the batched np.linalg.solve RBF coefficient computation
against the original per-element scipy.linalg.cho_factor/cho_solve loop.
"""

import time

import numpy as np
import pytest
import scipy.linalg as sla

from icon4py.model.common import type_alias as ta


# ============================================================
# Original (reference) implementation — per-element Cholesky
# ============================================================


def _original_solve_rbf_coefficients(
    z_rbfmat_np,
    rbf_offset_np,
    rhs_np,
    horizontal_start,
    num_zonal_meridional_components,
    rbf_offset_shape_full,
):
    num_elements = rbf_offset_np.shape[0]
    rbf_vec_coeff_np = [
        np.zeros(rbf_offset_shape_full, dtype=ta.wpfloat)
        for _ in range(num_zonal_meridional_components)
    ]
    for i in range(num_elements):
        valid_neighbors = np.where(rbf_offset_np[i, :] >= 0)[0]
        rbfmat_np = np.squeeze(z_rbfmat_np[np.ix_([i], valid_neighbors, valid_neighbors)])
        z_diag_np = sla.cho_factor(rbfmat_np)
        for j in range(num_zonal_meridional_components):
            rbf_vec_coeff_np[j][i + horizontal_start, valid_neighbors] = sla.cho_solve(
                z_diag_np, rhs_np[j][i, valid_neighbors]
            )
    return rbf_vec_coeff_np


# ============================================================
# Optimized implementation — batched np.linalg.solve
# ============================================================


def _batched_solve_rbf_coefficients(
    z_rbfmat_np,
    rbf_offset_np,
    rhs_np,
    horizontal_start,
    num_zonal_meridional_components,
    rbf_offset_shape_full,
):
    rbf_vec_coeff_np = [
        np.zeros(rbf_offset_shape_full, dtype=ta.wpfloat)
        for _ in range(num_zonal_meridional_components)
    ]
    n_valid = (rbf_offset_np >= 0).sum(axis=1)
    for nv in np.unique(n_valid):
        if nv == 0:
            continue
        group_idx = np.where(n_valid == nv)[0]
        valid_cols = np.arange(nv)
        mat_batch = z_rbfmat_np[np.ix_(group_idx, valid_cols, valid_cols)]
        for j in range(num_zonal_meridional_components):
            rhs_batch = rhs_np[j][np.ix_(group_idx, valid_cols)]
            sol = np.linalg.solve(mat_batch, rhs_batch[:, :, np.newaxis])[:, :, 0]
            rbf_vec_coeff_np[j][group_idx + horizontal_start, :nv] = sol
    return rbf_vec_coeff_np


# ============================================================
# Helper: generate SPD matrices with ICON-like connectivity
# ============================================================


def _make_rbf_test_data(rng, num_elements, max_neighbors, horizontal_start, num_components=2):
    """
    Generate symmetric positive definite matrices and RHS vectors
    mimicking the RBF interpolation linear systems.
    """
    total_size = horizontal_start + num_elements

    # rbf_offset: most elements have max_neighbors valid, some have fewer
    rbf_offset = rng.integers(0, 1000, size=(num_elements, max_neighbors))
    # Mark ~5% of last-column entries as invalid (-1)
    invalid_mask = rng.random(size=num_elements) < 0.05
    rbf_offset[invalid_mask, -1] = -1

    # z_rbfmat: SPD matrices (A = B @ B.T + diag)
    B = rng.uniform(-1.0, 1.0, size=(num_elements, max_neighbors, max_neighbors))
    z_rbfmat = np.matmul(B, B.transpose(0, 2, 1))
    z_rbfmat += np.eye(max_neighbors)[np.newaxis, :, :] * max_neighbors  # ensure well-conditioned

    # RHS vectors
    rhs = [
        rng.uniform(-1.0, 1.0, size=(num_elements, max_neighbors)) for _ in range(num_components)
    ]

    rbf_offset_shape_full = (total_size, max_neighbors)

    return z_rbfmat, rbf_offset, rhs, rbf_offset_shape_full


# ============================================================
# Tests
# ============================================================


@pytest.mark.level("unit")
class TestRbfBatchedSolve:
    """Test batched np.linalg.solve against per-element scipy Cholesky."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_matches_original(self, seed):
        rng = np.random.default_rng(seed)
        num_elements = 500
        max_neighbors = 6
        horizontal_start = 10
        num_components = 2

        z_rbfmat, rbf_offset, rhs, shape_full = _make_rbf_test_data(
            rng, num_elements, max_neighbors, horizontal_start, num_components
        )

        expected = _original_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )
        result = _batched_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )

        for j in range(num_components):
            np.testing.assert_allclose(result[j], expected[j], rtol=1e-10, atol=1e-13)

    def test_all_same_valid_count(self):
        """All elements have the same number of valid neighbors (common case)."""
        rng = np.random.default_rng(55)
        num_elements = 200
        max_neighbors = 6
        horizontal_start = 0
        num_components = 2

        z_rbfmat, rbf_offset, rhs, shape_full = _make_rbf_test_data(
            rng, num_elements, max_neighbors, horizontal_start, num_components
        )
        # Force all valid
        rbf_offset = np.abs(rbf_offset)

        expected = _original_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )
        result = _batched_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )

        for j in range(num_components):
            np.testing.assert_allclose(result[j], expected[j], rtol=1e-10, atol=1e-13)

    def test_mixed_valid_counts(self):
        """Elements have varying numbers of valid neighbors."""
        rng = np.random.default_rng(66)
        num_elements = 300
        max_neighbors = 6
        horizontal_start = 5
        num_components = 2

        z_rbfmat, rbf_offset, rhs, shape_full = _make_rbf_test_data(
            rng, num_elements, max_neighbors, horizontal_start, num_components
        )
        # Introduce more variation: some with 4, 5, or 6 valid neighbors
        for i in range(0, num_elements, 3):
            rbf_offset[i, -1] = -1
            rbf_offset[i, -2] = -1  # 4 valid
        for i in range(1, num_elements, 3):
            rbf_offset[i, -1] = -1  # 5 valid

        expected = _original_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )
        result = _batched_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )

        for j in range(num_components):
            np.testing.assert_allclose(result[j], expected[j], rtol=1e-10, atol=1e-13)

    def test_performance_improvement(self):
        """Verify batched solve is faster than per-element Cholesky."""
        rng = np.random.default_rng(77)
        num_elements = 10000
        max_neighbors = 6
        horizontal_start = 0
        num_components = 2

        z_rbfmat, rbf_offset, rhs, shape_full = _make_rbf_test_data(
            rng, num_elements, max_neighbors, horizontal_start, num_components
        )

        # Warm up
        _original_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )
        _batched_solve_rbf_coefficients(
            z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
        )

        n_runs = 3
        t_orig = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _original_solve_rbf_coefficients(
                z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
            )
            t_orig += time.perf_counter() - t0

        t_opt = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _batched_solve_rbf_coefficients(
                z_rbfmat, rbf_offset, rhs, horizontal_start, num_components, shape_full
            )
            t_opt += time.perf_counter() - t0

        speedup = t_orig / t_opt
        print(
            f"\nRBF batched solve speedup: {speedup:.1f}x"
            f"  (orig={t_orig/n_runs:.4f}s, opt={t_opt/n_runs:.4f}s)"
        )
        assert speedup > 2.0, f"Expected at least 2x speedup, got {speedup:.1f}x"

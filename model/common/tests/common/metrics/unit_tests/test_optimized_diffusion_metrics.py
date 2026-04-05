# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests comparing the vectorized compute_diffusion_intcoef_and_vertoffset
against the original scalar-loop version. Uses synthetic data with
monotonically decreasing height profiles to mimic ICON grid physics.
"""

import time

import numpy as np
import pytest


# ============================================================
# Original (reference) implementation — copied verbatim from
# the pre-optimization version of compute_diffusion_metrics.py
# ============================================================


def _original_compute_nbidx(k_range, z_mc, z_mc_off, nbidx, jc, nlev):
    for ind in range(3):
        jk_start = nlev - 1
        for jk in reversed(k_range):
            for jk1 in reversed(range(jk_start)):
                if (
                    z_mc[jc, jk] <= z_mc_off[jc, ind, jk1]
                    and z_mc[jc, jk] >= z_mc_off[jc, ind, jk1 + 1]
                ):
                    nbidx[jc, ind, jk] = jk1
                    jk_start = jk1 + 1
                    break
    return nbidx[jc, :, :]


def _original_compute_z_vintcoeff(k_range, z_mc, z_mc_off, z_vintcoeff, jc, nlev):
    for ind in range(3):
        jk_start = nlev - 1
        for jk in reversed(k_range):
            for jk1 in reversed(range(jk_start)):
                if (
                    z_mc[jc, jk] <= z_mc_off[jc, ind, jk1]
                    and z_mc[jc, jk] >= z_mc_off[jc, ind, jk1 + 1]
                ):
                    z_vintcoeff[jc, ind, jk] = (z_mc[jc, jk] - z_mc_off[jc, ind, jk1 + 1]) / (
                        z_mc_off[jc, ind, jk1] - z_mc_off[jc, ind, jk1 + 1]
                    )
                    jk_start = jk1 + 1
                    break
    return z_vintcoeff[jc, :, :]


def _original_compute_k_start_end(
    z_mc, max_nbhgt, maxslp_avg, maxhgtd_avg, c_owner_mask, thslp_zdiffu, thhgtd_zdiffu, nlev
):
    condition1 = np.logical_or(maxslp_avg >= thslp_zdiffu, maxhgtd_avg >= thhgtd_zdiffu)
    cell_mask = np.tile(np.where(condition1[:, nlev - 1], c_owner_mask, False), (nlev, 1)).T
    threshold = np.tile(max_nbhgt, (nlev, 1)).T
    owned_cell_above_threshold = np.logical_and(cell_mask, z_mc >= threshold)
    last_true_indices = nlev - 1 - np.argmax(owned_cell_above_threshold[:, ::-1], axis=1)
    kend = np.where(np.any(owned_cell_above_threshold, axis=1), last_true_indices + 1, 0)
    kstart = np.argmax(condition1, axis=1)
    kstart = np.where(kstart > kend, nlev, kstart)
    cell_index_mask = np.where(kend > kstart, True, False)
    return kstart, kend, cell_index_mask


def _original_compute_diffusion_intcoef_and_vertoffset(
    c2e2c,
    z_mc,
    max_nbhgt,
    c_owner_mask,
    maxslp_avg,
    maxhgtd_avg,
    thslp_zdiffu,
    thhgtd_zdiffu,
    cell_nudging,
    nlev,
):
    n_cells = c2e2c.shape[0]
    n_c2e2c = c2e2c.shape[1]
    z_mc_off = z_mc[c2e2c]
    nbidx = np.ones(shape=(n_cells, n_c2e2c, nlev), dtype=int)
    z_vintcoeff = np.zeros(shape=(n_cells, n_c2e2c, nlev))
    zd_vertoffset = np.zeros(shape=(n_cells, n_c2e2c, nlev), dtype=np.int32)
    zd_intcoef = np.zeros(shape=(n_cells, n_c2e2c, nlev))

    k_start, k_end, _ = _original_compute_k_start_end(
        z_mc,
        max_nbhgt,
        maxslp_avg,
        maxhgtd_avg,
        c_owner_mask,
        thslp_zdiffu,
        thhgtd_zdiffu,
        nlev,
    )

    for jc in range(cell_nudging, n_cells):
        kend = k_end[jc].item()
        kstart = k_start[jc].item()
        if kend > kstart:
            k_range = range(kstart, kend)
            nbidx[jc, :, :] = _original_compute_nbidx(k_range, z_mc, z_mc_off, nbidx, jc, nlev)
            z_vintcoeff[jc, :, :] = _original_compute_z_vintcoeff(
                k_range, z_mc, z_mc_off, z_vintcoeff, jc, nlev
            )
            zd_intcoef[jc, :, k_range] = z_vintcoeff[jc, :, k_range]
            zd_vertoffset[jc, :, k_range] = (
                nbidx[jc, :, k_range] - np.tile(np.array(k_range), (3, 1)).T
            )
    return zd_intcoef, zd_vertoffset


# ============================================================
# Optimized implementation — imported from production code
# ============================================================

from icon4py.model.common.metrics.compute_diffusion_metrics import (  # noqa: E402
    compute_diffusion_intcoef_and_vertoffset,
)


# ============================================================
# Helper: generate physically consistent test data
# ============================================================


def _make_diffusion_metrics_data(rng, n_cells, nlev, cell_nudging):
    """
    Generate synthetic height profiles and connectivity for testing.
    Heights are monotonically decreasing (higher levels = higher altitude).
    """
    # c2e2c: each cell has 3 neighbors
    c2e2c = rng.integers(0, n_cells, size=(n_cells, 3))

    # z_mc: monotonically decreasing height profiles (top of atmosphere to bottom)
    # Each cell has slightly different profile
    base_heights = np.linspace(50000.0, 0.0, nlev)  # 50km to 0
    perturbation = rng.uniform(-100.0, 100.0, size=(n_cells, nlev))
    z_mc = base_heights[np.newaxis, :] + perturbation
    # Ensure strict monotonically decreasing per cell
    z_mc = np.sort(z_mc, axis=1)[:, ::-1]

    # max_nbhgt: maximum neighbor height at the bottom level
    # Should be close to z_mc[:,nlev-1] but slightly higher for some cells
    max_nbhgt = z_mc[:, nlev - 1] + rng.uniform(0.0, 5000.0, size=n_cells)

    # c_owner_mask: all cells are owned
    c_owner_mask = np.ones(n_cells, dtype=bool)

    # maxslp_avg, maxhgtd_avg: condition metrics that determine which cells are active
    # Make ~50% of cells active by setting values above thresholds
    maxslp_avg = rng.uniform(0.0, 0.1, size=(n_cells, nlev))
    maxhgtd_avg = rng.uniform(0.0, 300.0, size=(n_cells, nlev))

    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0

    return (
        c2e2c,
        z_mc,
        max_nbhgt,
        c_owner_mask,
        maxslp_avg,
        maxhgtd_avg,
        thslp_zdiffu,
        thhgtd_zdiffu,
    )


# ============================================================
# Tests
# ============================================================


@pytest.mark.level("unit")
class TestComputeDiffusionIntcoef:
    """Test vectorized compute_diffusion_intcoef_and_vertoffset against original."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_matches_original(self, seed):
        rng = np.random.default_rng(seed)
        n_cells = 200
        nlev = 65
        cell_nudging = 5

        c2e2c, z_mc, max_nbhgt, c_owner_mask, maxslp_avg, maxhgtd_avg, thslp, thhgtd = (
            _make_diffusion_metrics_data(rng, n_cells, nlev, cell_nudging)
        )

        expected_intcoef, expected_vertoff = _original_compute_diffusion_intcoef_and_vertoffset(
            c2e2c,
            z_mc,
            max_nbhgt,
            c_owner_mask,
            maxslp_avg,
            maxhgtd_avg,
            thslp,
            thhgtd,
            cell_nudging,
            nlev,
        )
        result_intcoef, result_vertoff = compute_diffusion_intcoef_and_vertoffset(
            c2e2c,
            z_mc,
            max_nbhgt,
            c_owner_mask,
            maxslp_avg,
            maxhgtd_avg,
            thslp,
            thhgtd,
            cell_nudging,
            nlev,
            array_ns=np,
        )

        np.testing.assert_allclose(result_intcoef, expected_intcoef, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(result_vertoff, expected_vertoff)

    def test_no_active_cells(self):
        """All cells have empty k_range — should produce all-zero output."""
        rng = np.random.default_rng(7)
        n_cells = 50
        nlev = 20
        cell_nudging = 0

        c2e2c = rng.integers(0, n_cells, size=(n_cells, 3))
        z_mc = np.sort(rng.uniform(0, 50000, (n_cells, nlev)), axis=1)[:, ::-1]
        max_nbhgt = z_mc[:, 0] + 1000.0  # above all heights → no cells active
        c_owner_mask = np.ones(n_cells, dtype=bool)
        maxslp_avg = np.zeros((n_cells, nlev))
        maxhgtd_avg = np.zeros((n_cells, nlev))

        result_intcoef, result_vertoff = compute_diffusion_intcoef_and_vertoffset(
            c2e2c,
            z_mc,
            max_nbhgt,
            c_owner_mask,
            maxslp_avg,
            maxhgtd_avg,
            0.02,
            125.0,
            cell_nudging,
            nlev,
            array_ns=np,
        )

        np.testing.assert_array_equal(result_intcoef, 0.0)
        np.testing.assert_array_equal(result_vertoff, 0)

    def test_performance_improvement(self):
        """Verify vectorized version is faster on a moderately sized grid."""
        rng = np.random.default_rng(77)
        n_cells = 2000
        nlev = 65
        cell_nudging = 5

        c2e2c, z_mc, max_nbhgt, c_owner_mask, maxslp_avg, maxhgtd_avg, thslp, thhgtd = (
            _make_diffusion_metrics_data(rng, n_cells, nlev, cell_nudging)
        )

        # Warm up
        _original_compute_diffusion_intcoef_and_vertoffset(
            c2e2c,
            z_mc,
            max_nbhgt,
            c_owner_mask,
            maxslp_avg,
            maxhgtd_avg,
            thslp,
            thhgtd,
            cell_nudging,
            nlev,
        )
        compute_diffusion_intcoef_and_vertoffset(
            c2e2c,
            z_mc,
            max_nbhgt,
            c_owner_mask,
            maxslp_avg,
            maxhgtd_avg,
            thslp,
            thhgtd,
            cell_nudging,
            nlev,
            array_ns=np,
        )

        n_runs = 3
        t_orig = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _original_compute_diffusion_intcoef_and_vertoffset(
                c2e2c,
                z_mc,
                max_nbhgt,
                c_owner_mask,
                maxslp_avg,
                maxhgtd_avg,
                thslp,
                thhgtd,
                cell_nudging,
                nlev,
            )
            t_orig += time.perf_counter() - t0

        t_opt = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            compute_diffusion_intcoef_and_vertoffset(
                c2e2c,
                z_mc,
                max_nbhgt,
                c_owner_mask,
                maxslp_avg,
                maxhgtd_avg,
                thslp,
                thhgtd,
                cell_nudging,
                nlev,
                array_ns=np,
            )
            t_opt += time.perf_counter() - t0

        speedup = t_orig / t_opt
        print(
            f"\ncompute_diffusion_intcoef_and_vertoffset speedup: {speedup:.1f}x"
            f"  (orig={t_orig/n_runs:.4f}s, opt={t_opt/n_runs:.4f}s)"
        )
        assert speedup > 5.0, f"Expected at least 5x speedup, got {speedup:.1f}x"

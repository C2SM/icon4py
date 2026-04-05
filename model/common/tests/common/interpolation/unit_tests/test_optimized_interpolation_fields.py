# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests comparing optimized interpolation field implementations against the original
scalar-loop versions. Each test generates random connectivity and data, runs both
implementations, and asserts numerical equivalence.
"""

import time

import numpy as np
import pytest

from icon4py.model.common.grid.gridfile import GridFile


MISSING = GridFile.INVALID_INDEX


# ============================================================
# Original (reference) implementations — copied verbatim
# ============================================================


def _original_compute_cells_aw_verts(
    dual_area, edge_vert_length, edge_cell_length, v2e, e2v, v2c, e2c, horizontal_start
):
    cells_aw_verts = np.zeros(v2e.shape)
    for jv in range(horizontal_start, cells_aw_verts.shape[0]):
        for je in range(v2e.shape[1]):
            if v2e[jv, je] == MISSING or (je > 0 and v2e[jv, je] == v2e[jv, je - 1]):
                continue
            ile = v2e[jv, je]
            idx_ve = 0 if e2v[ile, 0] == jv else 1
            cell_offset_idx_0 = e2c[ile, 0]
            cell_offset_idx_1 = e2c[ile, 1]
            for jc in range(v2e.shape[1]):
                if v2c[jv, jc] == MISSING or (jc > 0 and v2c[jv, jc] == v2c[jv, jc - 1]):
                    continue
                if cell_offset_idx_0 == v2c[jv, jc]:
                    cells_aw_verts[jv, jc] = (
                        cells_aw_verts[jv, jc]
                        + 0.5
                        / dual_area[jv]
                        * edge_vert_length[ile, idx_ve]
                        * edge_cell_length[ile, 0]
                    )
                elif cell_offset_idx_1 == v2c[jv, jc]:
                    cells_aw_verts[jv, jc] = (
                        cells_aw_verts[jv, jc]
                        + 0.5
                        / dual_area[jv]
                        * edge_vert_length[ile, idx_ve]
                        * edge_cell_length[ile, 1]
                    )
    return cells_aw_verts


def _original_compute_lsq_pseudoinv(
    cell_owner_mask,
    lsq_pseudoinv,
    z_lsq_mat_c,
    lsq_weights_c,
    start_idx,
    min_rlcell_int,
    lsq_dim_unk,
    lsq_dim_c,
):
    for jjb in range(lsq_dim_c):
        for jjk in range(lsq_dim_unk):
            for jc in range(start_idx, min_rlcell_int):
                if cell_owner_mask[jc]:
                    u, s, v_t = np.linalg.svd(z_lsq_mat_c[jc, :, :])
                    lsq_pseudoinv[jc, :lsq_dim_unk, jjb] = (
                        lsq_pseudoinv[jc, :lsq_dim_unk, jjb]
                        + v_t[jjk, :lsq_dim_unk] / s[jjk] * u[jjb, jjk] * lsq_weights_c[jc, jjb]
                    )
    return lsq_pseudoinv


def _original_create_inverse_neighbor_index(source_offset, inverse_offset):
    inv_neighbor_idx = MISSING * np.ones(inverse_offset.shape, dtype=np.int32)
    for jc in range(inverse_offset.shape[0]):
        for i in range(inverse_offset.shape[1]):
            if inverse_offset[jc, i] >= 0:
                inverse_nn = np.argwhere(source_offset[inverse_offset[jc, i], :] == jc)
                inv_neighbor_idx[jc, i] = inverse_nn[0, 0] if len(inverse_nn) > 0 else MISSING
    return inv_neighbor_idx


# ============================================================
# New (optimized) implementations — imported from production code
# ============================================================

from icon4py.model.common.interpolation.interpolation_fields import (
    _create_inverse_neighbor_index,
    compute_cells_aw_verts,
    compute_lsq_pseudoinv,
)


# ============================================================
# Helper: generate ICON-like connectivity for a small test grid
# ============================================================


def _make_icon_like_connectivity(rng, num_verts, num_edges, num_cells, edges_per_vert=6):
    """
    Build synthetic connectivity tables that are structurally consistent
    with an ICON triangular grid (with some MISSING entries).
    """
    # e2v: each edge connects 2 vertices
    e2v = rng.integers(0, num_verts, size=(num_edges, 2))
    # make sure endpoints differ
    for i in range(num_edges):
        while e2v[i, 0] == e2v[i, 1]:
            e2v[i, 1] = rng.integers(0, num_verts)

    # e2c: each edge has 2 adjacent cells
    e2c = rng.integers(0, num_cells, size=(num_edges, 2))

    # v2e: each vertex has up to edges_per_vert edges, some MISSING
    v2e = np.full((num_verts, edges_per_vert), MISSING, dtype=np.int64)
    for jv in range(num_verts):
        incident = np.where((e2v[:, 0] == jv) | (e2v[:, 1] == jv))[0]
        # deduplicate
        incident = np.unique(incident)
        n = min(len(incident), edges_per_vert)
        v2e[jv, :n] = incident[:n]

    # v2c: cells around each vertex — gather from e2c via v2e
    v2c = np.full((num_verts, edges_per_vert), MISSING, dtype=np.int64)
    for jv in range(num_verts):
        cells = set()
        for je in range(edges_per_vert):
            if v2e[jv, je] != MISSING:
                cells.add(e2c[v2e[jv, je], 0])
                cells.add(e2c[v2e[jv, je], 1])
        cells_list = sorted(cells)
        n = min(len(cells_list), edges_per_vert)
        v2c[jv, :n] = cells_list[:n]

    return v2e, e2v, v2c, e2c


# ============================================================
# Tests
# ============================================================


@pytest.mark.level("unit")
class TestComputeCellsAwVerts:
    """Test vectorized compute_cells_aw_verts against original loop version."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_matches_original(self, seed):
        rng = np.random.default_rng(seed)
        num_verts, num_edges, num_cells = 500, 1200, 800
        edges_per_vert = 6
        horizontal_start = 5

        v2e, e2v, v2c, e2c = _make_icon_like_connectivity(
            rng, num_verts, num_edges, num_cells, edges_per_vert
        )
        dual_area = rng.uniform(1.0, 10.0, size=num_verts)
        edge_vert_length = rng.uniform(0.1, 5.0, size=(num_edges, 2))
        edge_cell_length = rng.uniform(0.1, 5.0, size=(num_edges, 2))

        no_exchange = lambda x: None

        expected = _original_compute_cells_aw_verts(
            dual_area, edge_vert_length, edge_cell_length, v2e, e2v, v2c, e2c, horizontal_start
        )
        result = compute_cells_aw_verts(
            dual_area,
            edge_vert_length,
            edge_cell_length,
            v2e,
            e2v,
            v2c,
            e2c,
            horizontal_start,
            exchange=no_exchange,
            array_ns=np,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)

    def test_performance_improvement(self):
        """Verify optimized version is faster on a moderately sized grid."""
        rng = np.random.default_rng(77)
        num_verts, num_edges, num_cells = 5000, 12000, 8000
        edges_per_vert = 6
        horizontal_start = 10

        v2e, e2v, v2c, e2c = _make_icon_like_connectivity(
            rng, num_verts, num_edges, num_cells, edges_per_vert
        )
        dual_area = rng.uniform(1.0, 10.0, size=num_verts)
        edge_vert_length = rng.uniform(0.1, 5.0, size=(num_edges, 2))
        edge_cell_length = rng.uniform(0.1, 5.0, size=(num_edges, 2))

        no_exchange = lambda x: None

        # Warm up
        _original_compute_cells_aw_verts(
            dual_area, edge_vert_length, edge_cell_length, v2e, e2v, v2c, e2c, horizontal_start
        )
        compute_cells_aw_verts(
            dual_area,
            edge_vert_length,
            edge_cell_length,
            v2e,
            e2v,
            v2c,
            e2c,
            horizontal_start,
            exchange=no_exchange,
            array_ns=np,
        )

        n_runs = 3
        t_orig = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _original_compute_cells_aw_verts(
                dual_area, edge_vert_length, edge_cell_length, v2e, e2v, v2c, e2c, horizontal_start
            )
            t_orig += time.perf_counter() - t0

        t_opt = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            compute_cells_aw_verts(
                dual_area,
                edge_vert_length,
                edge_cell_length,
                v2e,
                e2v,
                v2c,
                e2c,
                horizontal_start,
                exchange=no_exchange,
                array_ns=np,
            )
            t_opt += time.perf_counter() - t0

        speedup = t_orig / t_opt
        print(
            f"\ncompute_cells_aw_verts speedup: {speedup:.1f}x  (orig={t_orig/n_runs:.4f}s, opt={t_opt/n_runs:.4f}s)"
        )
        assert speedup > 2.0, f"Expected at least 2x speedup, got {speedup:.1f}x"


@pytest.mark.level("unit")
class TestComputeLsqPseudoinv:
    """Test batch-SVD compute_lsq_pseudoinv against original triple-loop version."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_matches_original(self, seed):
        rng = np.random.default_rng(seed)
        n_cells = 200
        lsq_dim_unk = 2
        lsq_dim_c = 3
        start_idx = 5

        cell_owner_mask = np.ones(n_cells, dtype=bool)
        cell_owner_mask[:start_idx] = False
        # randomly mark ~10% as non-owned
        cell_owner_mask[rng.choice(np.arange(start_idx, n_cells), size=20, replace=False)] = False

        # z_lsq_mat_c: needs to be non-singular for SVD
        z_lsq_mat_c = rng.uniform(-1.0, 1.0, size=(n_cells, lsq_dim_c, lsq_dim_c))
        # Make matrices well-conditioned by adding identity
        for jc in range(n_cells):
            z_lsq_mat_c[jc] += 3.0 * np.eye(lsq_dim_c)

        lsq_weights_c = rng.uniform(0.1, 1.0, size=(n_cells, lsq_dim_c))

        lsq_pseudoinv_orig = np.zeros((n_cells, lsq_dim_unk, lsq_dim_c))
        lsq_pseudoinv_new = np.zeros((n_cells, lsq_dim_unk, lsq_dim_c))

        expected = _original_compute_lsq_pseudoinv(
            cell_owner_mask,
            lsq_pseudoinv_orig.copy(),
            z_lsq_mat_c.copy(),
            lsq_weights_c,
            start_idx,
            n_cells,
            lsq_dim_unk,
            lsq_dim_c,
        )
        result = compute_lsq_pseudoinv(
            cell_owner_mask,
            lsq_pseudoinv_new.copy(),
            z_lsq_mat_c.copy(),
            lsq_weights_c,
            start_idx,
            n_cells,
            lsq_dim_unk,
            lsq_dim_c,
            array_ns=np,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-13)

    def test_performance_improvement(self):
        """Verify batch SVD is faster than per-cell SVD in triple loop."""
        rng = np.random.default_rng(77)
        n_cells = 2000
        lsq_dim_unk = 2
        lsq_dim_c = 3
        start_idx = 5

        cell_owner_mask = np.ones(n_cells, dtype=bool)
        cell_owner_mask[:start_idx] = False

        z_lsq_mat_c = rng.uniform(-1.0, 1.0, size=(n_cells, lsq_dim_c, lsq_dim_c))
        for jc in range(n_cells):
            z_lsq_mat_c[jc] += 3.0 * np.eye(lsq_dim_c)
        lsq_weights_c = rng.uniform(0.1, 1.0, size=(n_cells, lsq_dim_c))

        # Warm up
        _original_compute_lsq_pseudoinv(
            cell_owner_mask,
            np.zeros((n_cells, lsq_dim_unk, lsq_dim_c)),
            z_lsq_mat_c.copy(),
            lsq_weights_c,
            start_idx,
            n_cells,
            lsq_dim_unk,
            lsq_dim_c,
        )
        compute_lsq_pseudoinv(
            cell_owner_mask,
            np.zeros((n_cells, lsq_dim_unk, lsq_dim_c)),
            z_lsq_mat_c.copy(),
            lsq_weights_c,
            start_idx,
            n_cells,
            lsq_dim_unk,
            lsq_dim_c,
            array_ns=np,
        )

        n_runs = 3
        t_orig = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _original_compute_lsq_pseudoinv(
                cell_owner_mask,
                np.zeros((n_cells, lsq_dim_unk, lsq_dim_c)),
                z_lsq_mat_c.copy(),
                lsq_weights_c,
                start_idx,
                n_cells,
                lsq_dim_unk,
                lsq_dim_c,
            )
            t_orig += time.perf_counter() - t0

        t_opt = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            compute_lsq_pseudoinv(
                cell_owner_mask,
                np.zeros((n_cells, lsq_dim_unk, lsq_dim_c)),
                z_lsq_mat_c.copy(),
                lsq_weights_c,
                start_idx,
                n_cells,
                lsq_dim_unk,
                lsq_dim_c,
                array_ns=np,
            )
            t_opt += time.perf_counter() - t0

        speedup = t_orig / t_opt
        print(
            f"\ncompute_lsq_pseudoinv speedup: {speedup:.1f}x  (orig={t_orig/n_runs:.4f}s, opt={t_opt/n_runs:.4f}s)"
        )
        assert speedup > 5.0, f"Expected at least 5x speedup, got {speedup:.1f}x"


@pytest.mark.level("unit")
class TestCreateInverseNeighborIndex:
    """Test vectorized _create_inverse_neighbor_index against original double-loop version."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_matches_original(self, seed):
        rng = np.random.default_rng(seed)
        n_cells = 500
        n_neighbors = 4  # C2E2C0 has 4 columns

        # Build a c2e2c0-like connectivity: first column is self, rest are neighbors
        c2e2c0 = np.full((n_cells, n_neighbors), MISSING, dtype=np.int64)
        c2e2c0[:, 0] = np.arange(n_cells)
        for i in range(1, n_neighbors):
            c2e2c0[:, i] = rng.integers(0, n_cells, size=n_cells)

        # Some entries should be MISSING
        mask = rng.random(size=(n_cells, n_neighbors)) < 0.05
        mask[:, 0] = False  # keep self-reference
        c2e2c0[mask] = MISSING

        expected = _original_create_inverse_neighbor_index(c2e2c0, c2e2c0)
        result = _create_inverse_neighbor_index(c2e2c0, c2e2c0, array_ns=np)

        np.testing.assert_array_equal(result, expected)

    def test_performance_improvement(self):
        """Verify vectorized version is faster."""
        rng = np.random.default_rng(77)
        n_cells = 5000
        n_neighbors = 4

        c2e2c0 = np.full((n_cells, n_neighbors), MISSING, dtype=np.int64)
        c2e2c0[:, 0] = np.arange(n_cells)
        for i in range(1, n_neighbors):
            c2e2c0[:, i] = rng.integers(0, n_cells, size=n_cells)

        # Warm up
        _original_create_inverse_neighbor_index(c2e2c0, c2e2c0)
        _create_inverse_neighbor_index(c2e2c0, c2e2c0, array_ns=np)

        n_runs = 3
        t_orig = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _original_create_inverse_neighbor_index(c2e2c0, c2e2c0)
            t_orig += time.perf_counter() - t0

        t_opt = 0.0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _create_inverse_neighbor_index(c2e2c0, c2e2c0, array_ns=np)
            t_opt += time.perf_counter() - t0

        speedup = t_orig / t_opt
        print(
            f"\n_create_inverse_neighbor_index speedup: {speedup:.1f}x  (orig={t_orig/n_runs:.4f}s, opt={t_opt/n_runs:.4f}s)"
        )
        assert speedup > 2.0, f"Expected at least 2x speedup, got {speedup:.1f}x"

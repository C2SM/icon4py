# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import rng
from icon4py.model.common import dimension as dims


def _call(cell_id, k_id, bin_id, step, backend=None):
    op = rng.counter_hash01.with_backend(backend) if backend is not None else rng.counter_hash01
    out = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_id.shape[0], k_id.shape[0]), dtype=np.float64)
    )
    op(
        cell_id,
        k_id,
        gtx.int64(bin_id),
        gtx.int64(step),
        gtx.int64(rng.C_CELL),
        gtx.int64(rng.C_K),
        gtx.int64(rng.C_BIN),
        gtx.int64(rng.C_STEP),
        gtx.int64(rng.M31),
        gtx.int64(rng.A1),
        gtx.int64(rng.A2),
        gtx.int64(rng.A3),
        out=out,
        offset_provider={},
    )
    return out.asnumpy()


class TestBitExactness:
    @pytest.mark.parametrize("bin_id,step", [(0, 0), (7, 1234), (3, 999999)])
    def test_embedded_matches_numpy_replica(self, bin_id, step):
        ncells, nlev = 64, 17
        cell_id = gtx.as_field((dims.CellDim,), np.arange(ncells, dtype=np.int64))
        k_id = gtx.as_field((dims.KDim,), np.arange(nlev, dtype=np.int64))

        got = _call(cell_id, k_id, bin_id, step)
        expected = rng.counter_hash01_numpy(np.arange(ncells), np.arange(nlev), bin_id, step)
        assert np.array_equal(got, expected)

    def test_gtfn_cpu_matches_numpy_replica(self):
        """Optional per the task brief ("gtfn optional -- small compile")
        -- included here since, unlike the 40-bin collection kernel
        spike, this operator's signature is tiny and compiles fast."""
        ncells, nlev = 32, 11
        cell_id = gtx.as_field((dims.CellDim,), np.arange(ncells, dtype=np.int64))
        k_id = gtx.as_field((dims.KDim,), np.arange(nlev, dtype=np.int64))

        got = _call(cell_id, k_id, bin_id=7, step=1234, backend=run_gtfn_cached)
        expected = rng.counter_hash01_numpy(np.arange(ncells), np.arange(nlev), 7, 1234)
        assert np.array_equal(got, expected)

    def test_different_bin_or_step_gives_different_output(self):
        ncells, nlev = 16, 5
        cell_id = gtx.as_field((dims.CellDim,), np.arange(ncells, dtype=np.int64))
        k_id = gtx.as_field((dims.KDim,), np.arange(nlev, dtype=np.int64))
        a = _call(cell_id, k_id, bin_id=0, step=0)
        b = _call(cell_id, k_id, bin_id=1, step=0)
        c = _call(cell_id, k_id, bin_id=0, step=1)
        assert not np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestNumpyReplicaRange:
    def test_output_in_unit_interval(self):
        got = rng.counter_hash01_numpy(np.arange(500), np.arange(61), bin_id=3, step=42)
        assert np.all(got >= 0.0)
        assert np.all(got < 1.0)

    def test_quality_bar(self):
        """Reproduces spike_e_counter_rng.py's quality assertions at the
        SAME grid scale that spike used (NCELLS=4096, NLEV=61, F5 SS6's
        common.py constants) -- the lag-1/histogram metrics are noisy at
        small sample sizes, so this uses the already-validated scale
        rather than a smaller, noisier one."""
        ncells, nlev = 4096, 61
        samples = [
            rng.counter_hash01_numpy(np.arange(ncells), np.arange(nlev), b, s).ravel()
            for b in range(4)
            for s in range(4)
        ]
        sample = np.concatenate(samples)
        mean, var = sample.mean(), sample.var()
        grid = rng.counter_hash01_numpy(np.arange(ncells), np.arange(nlev), 7, 1234)
        lag1_cell = np.corrcoef(grid[:-1].ravel(), grid[1:].ravel())[0, 1]
        lag1_k = np.corrcoef(grid[:, :-1].ravel(), grid[:, 1:].ravel())[0, 1]
        hist = np.histogram(sample, bins=16, range=(0.0, 1.0))[0] / sample.size
        hist_dev = np.abs(hist - 1.0 / 16).max()

        assert abs(mean - 0.5) < 0.005
        assert abs(var - 1.0 / 12.0) < 0.005
        assert abs(lag1_cell) < 0.01
        assert abs(lag1_k) < 0.01
        assert hist_dev < 0.01


class TestQuadraticCollision:
    """Verifies the module docstring's 2-to-1 collision claim about the
    quadratic mixing round in isolation (not the full hash chain)."""

    def test_pair_collision(self):
        m31 = rng.M31

        def f(x):
            return (x * x + x + 1) % m31

        rng_np = np.random.default_rng(0)
        for _ in range(50):
            x1 = int(rng_np.integers(0, m31))
            x2 = (m31 - 1 - x1) % m31
            assert f(x1) == f(x2)

    def test_fixed_point(self):
        m31 = rng.M31
        fixed = ((m31 - 1) * pow(2, -1, m31)) % m31
        assert (2 * fixed) % m31 == (m31 - 1) % m31

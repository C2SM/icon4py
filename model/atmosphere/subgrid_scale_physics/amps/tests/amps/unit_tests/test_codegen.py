# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for `codegen/generate.py` + `codegen/templates.py`, using the
committed `core/generated/axpy_demo.py` (produced by
`generate.emit_module("axpy_demo", templates.build_axpy_per_bin,
nbins=10)`, see its own module for the exact params). Per generate.py's
module docstring, tests never call `emit_module` (which writes to disk)
-- only the read-only `check_regenerated` and `load_generated_module`.
"""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.codegen import generate, templates
from icon4py.model.common import dimension as dims


AXPY_DEMO_NBINS = 10  # must match the params used to produce the committed file


class TestChunkBins:
    def test_exact_multiple(self):
        assert templates.chunk_bins(16, chunk_size=8) == [(0, 8), (8, 16)]

    def test_remainder(self):
        assert templates.chunk_bins(10, chunk_size=8) == [(0, 8), (8, 10)]

    def test_smaller_than_chunk_size(self):
        assert templates.chunk_bins(3, chunk_size=8) == [(0, 3)]

    def test_never_exceeds_chunk_size(self):
        for lo, hi in templates.chunk_bins(40, chunk_size=8):
            assert hi - lo <= 8

    def test_covers_every_bin_exactly_once(self):
        chunks = templates.chunk_bins(23, chunk_size=8)
        covered = [b for lo, hi in chunks for b in range(lo, hi)]
        assert covered == list(range(23))

    def test_invalid_nbins_raises(self):
        with pytest.raises(ValueError):
            templates.chunk_bins(0)

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            templates.chunk_bins(5, chunk_size=0)


class TestDriftGuard:
    def test_axpy_demo_matches_committed_file(self):
        """The drift guard: re-running the SAME builder+params used to
        produce core/generated/axpy_demo.py must reproduce it exactly."""
        generate.check_regenerated("axpy_demo", templates.build_axpy_per_bin, nbins=AXPY_DEMO_NBINS)

    def test_mismatched_params_are_detected(self):
        """Sanity check on the guard itself: a builder call with
        DIFFERENT params than what's committed must be flagged, proving
        check_regenerated() is not a vacuous no-op."""
        with pytest.raises(AssertionError):
            generate.check_regenerated(
                "axpy_demo", templates.build_axpy_per_bin, nbins=AXPY_DEMO_NBINS + 1
            )

    def test_missing_committed_file_raises(self):
        with pytest.raises(AssertionError):
            generate.check_regenerated("does_not_exist_ever", templates.build_axpy_per_bin, nbins=3)


class TestGeneratedModuleChunking:
    """Proves the committed axpy_demo.py genuinely SPLITS into multiple
    field_operators (one per <=8-bin group), not one monolithic one."""

    def test_chunk_boundaries_match_source(self):
        chunks = templates.chunk_bins(AXPY_DEMO_NBINS, chunk_size=8)
        assert chunks == [(0, 8), (8, 10)]

    def test_module_has_one_field_operator_per_chunk(self):
        module = generate.load_generated_module("axpy_demo")
        for lo, hi in templates.chunk_bins(AXPY_DEMO_NBINS, chunk_size=8):
            assert hasattr(module, templates.axpy_chunk_name(lo, hi))


class TestEmbeddedExecutionMatchesNumpy:
    def test_all_chunks(self):
        module = generate.load_generated_module("axpy_demo")
        coeffs = templates.default_axpy_coeffs(AXPY_DEMO_NBINS)

        rng = np.random.default_rng(42)
        ncells, nlev = 6, 4

        for lo, hi in templates.chunk_bins(AXPY_DEMO_NBINS, chunk_size=8):
            op = getattr(module, templates.axpy_chunk_name(lo, hi))
            bins = list(range(lo, hi))

            x_np = {b: rng.uniform(-5.0, 5.0, size=(ncells, nlev)) for b in bins}
            y_np = {b: rng.uniform(-5.0, 5.0, size=(ncells, nlev)) for b in bins}
            kwargs = {}
            for b in bins:
                kwargs[f"x_{b:02d}"] = gtx.as_field((dims.CellDim, dims.KDim), x_np[b])
                kwargs[f"y_{b:02d}"] = gtx.as_field((dims.CellDim, dims.KDim), y_np[b])

            outs = tuple(
                gtx.as_field((dims.CellDim, dims.KDim), np.zeros((ncells, nlev))) for _ in bins
            )
            op(**kwargs, out=outs, offset_provider={})

            for out_field, b in zip(outs, bins, strict=True):
                expected = coeffs[b] * x_np[b] + y_np[b]
                assert np.array_equal(out_field.asnumpy(), expected), f"bin {b} mismatch"

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the savepoint archive inspector."""

from __future__ import annotations

import types

import numpy as np
import pytest
from inspect_savepoints import ArchiveExplorer, SavepointRef, summarize


def _summarize(values: np.ndarray):
    return summarize(values, savepoint="sp", date=None, field="f", component=None)


def test_summarize_reports_range_and_mean():
    stats = _summarize(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert (stats.min, stats.max, stats.mean) == (1.0, 4.0, 2.5)
    assert stats.shape == (2, 2)
    assert stats.nonzero_fraction == 1.0
    assert not stats.all_zero


def test_summarize_flags_an_all_zero_field():
    # The question the tool exists to answer: did this field get written at all?
    assert _summarize(np.zeros((3, 3))).all_zero


def test_summarize_excludes_non_finite_values_from_the_range():
    # A single NaN would otherwise swallow the min and max of the whole field.
    stats = _summarize(np.array([1.0, np.nan, 3.0, np.inf]))

    assert (stats.min, stats.max) == (1.0, 3.0)
    assert stats.n_nonfinite == 2


def _explorer(arrays: dict[tuple[int, str], np.ndarray]) -> ArchiveExplorer:
    # '__init__' opens a real archive; 'diff_stats' only needs these two attributes.
    explorer = object.__new__(ArchiveExplorer)
    explorer._serializer = types.SimpleNamespace(read=lambda field, sp: arrays[(sp, field)])
    explorer._raw = (0, 1)
    return explorer


def test_diff_stats_summarizes_the_change_between_two_savepoints():
    before = SavepointRef(0, "diffusion-init", "2001-01-01T00:00:04.000")
    after = SavepointRef(1, "diffusion-exit", "2001-01-01T00:00:04.000")
    explorer = _explorer({(0, "theta_v"): np.zeros((2, 2)), (1, "theta_v"): np.full((2, 2), 0.5)})

    [stats] = explorer.diff_stats(before, after, "theta_v")

    assert (stats.min, stats.max, stats.mean) == (0.5, 0.5, 0.5)
    # the two endpoints are named, so the row says what was compared
    assert stats.savepoint == "diffusion-init -> diffusion-exit"
    # equal endpoints collapse rather than repeating themselves
    assert stats.date == "2001-01-01T00:00:04.000"


def test_diff_stats_rejects_savepoints_of_different_shape():
    before = SavepointRef(0, "a", None)
    after = SavepointRef(1, "b", None)
    explorer = _explorer({(0, "vn"): np.zeros((2, 2)), (1, "vn"): np.zeros((3, 3))})

    with pytest.raises(ValueError, match="Shape mismatch"):
        explorer.diff_stats(before, after, "vn")

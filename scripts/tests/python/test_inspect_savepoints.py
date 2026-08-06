# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the savepoint archive inspector."""

from __future__ import annotations

import numpy as np
from inspect_savepoints import summarize


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

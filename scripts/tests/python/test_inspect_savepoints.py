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
import typer
from inspect_savepoints import (
    ArchiveExplorer,
    SavepointRef,
    _pair_by_metainfo,
    distinguishing_keys,
    step_label,
    summarize,
)


def _summarize(values: np.ndarray):
    return summarize(values, savepoint="sp", date=None, field="f", component=None)


def _ref(index: int, name: str, **metainfo) -> SavepointRef:
    return SavepointRef(index, name, tuple(sorted(metainfo.items())))


def test_summarize_reports_range_and_mean():
    stats = _summarize(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert (stats.min, stats.max, stats.mean) == (1.0, 4.0, 2.5)
    assert stats.shape == (2, 2)
    assert stats.nonzero_fraction == 1.0


def test_summarize_reports_the_shape_as_serialized():
    # ICON writes most fields with singleton dimensions; the row shows what is in the
    # file rather than a compacted version of it, and the statistics do not depend on it
    stats = _summarize(np.ones((3, 1, 2)))

    assert stats.shape == (3, 1, 2)
    assert (stats.min, stats.max, stats.mean) == (1.0, 1.0, 1.0)


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
    before = _ref(0, "diffusion-init", date="2001-01-01T00:00:04.000")
    after = _ref(1, "diffusion-exit", date="2001-01-01T00:00:04.000")
    explorer = _explorer({(0, "theta_v"): np.zeros((2, 2)), (1, "theta_v"): np.full((2, 2), 0.5)})

    [stats] = explorer.diff_stats(before, after, "theta_v")

    assert (stats.min, stats.max, stats.mean) == (0.5, 0.5, 0.5)
    # the two endpoints are named, so the row says what was compared
    assert stats.savepoint == "diffusion-init -> diffusion-exit"
    # equal endpoints collapse rather than repeating themselves
    assert stats.date == "2001-01-01T00:00:04.000"


def test_diff_stats_rejects_savepoints_of_different_shape():
    before = _ref(0, "a")
    after = _ref(1, "b")
    explorer = _explorer({(0, "vn"): np.zeros((2, 2)), (1, "vn"): np.zeros((3, 3))})

    with pytest.raises(ValueError, match="Shape mismatch"):
        explorer.diff_stats(before, after, "vn")


def test_diff_stats_reports_the_step_of_the_compared_savepoints():
    before = _ref(0, "solve-nonhydro-init", date="d", dyn_timestep=2, istep=1)
    after = _ref(1, "solve-nonhydro-exit", date="d", dyn_timestep=2, istep=1)
    explorer = _explorer({(0, "vn"): np.zeros((2, 2)), (1, "vn"): np.ones((2, 2))})

    [stats] = explorer.diff_stats(before, after, "vn", keys=("dyn_timestep", "istep"))

    assert stats.step == "dyn_timestep=2 istep=1"


def _substeps(name: str, **extra) -> list[SavepointRef]:
    # what ICON writes for one date: one savepoint per dynamical substep and stage
    return [
        _ref(index, name, date="d", dyn_timestep=step, istep=stage, **extra)
        for index, (step, stage) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)])
    ]


def test_distinguishing_keys_ignores_what_a_savepoint_shares():
    assert distinguishing_keys(_substeps("solve-nonhydro-init", dtime=0.02)) == (
        "dyn_timestep",
        "istep",
    )


def test_step_label_renders_only_the_requested_keys():
    reference = _ref(0, "sp", date="d", dyn_timestep=3, istep=2)

    assert step_label(reference, ("dyn_timestep", "istep")) == "dyn_timestep=3 istep=2"


def test_pairing_matches_savepoints_of_the_same_substep():
    # a date is not an identifier: all four of these carry the same one, and pairing on
    # it alone would compare every 'init' against the last 'exit'
    initial = _substeps("solve-nonhydro-init", linit=False)
    final = _substeps("solve-nonhydro-exit", prep_adv=False)

    pairs = _pair_by_metainfo(initial, final)

    assert [(before.index, after.index) for before, after in pairs] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]


def test_pairing_keeps_only_the_steps_both_savepoints_recorded():
    initial = _substeps("velocity-tendencies-init")
    final = [
        reference for reference in _substeps("velocity-tendencies-exit") if reference.index != 2
    ]

    pairs = _pair_by_metainfo(initial, final)

    assert [before.index for before, _ in pairs] == [0, 1, 3]


def test_pairing_refuses_savepoints_it_cannot_tell_apart():
    # 'solve-nonhydro-final' is written once per substep and does not record 'istep',
    # so nothing says which of the two stages of 'solve-nonhydro-exit' it belongs to
    exits = _substeps("solve-nonhydro-exit")
    finals = [
        _ref(index, "solve-nonhydro-final", date="d", dyn_timestep=step)
        for index, step in enumerate([1, 2])
    ]

    with pytest.raises(typer.BadParameter, match="cannot be paired"):
        _pair_by_metainfo(exits, finals)

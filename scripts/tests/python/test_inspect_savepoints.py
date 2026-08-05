# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the savepoint archive inspector."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest
import typer
from inspect_savepoints import (
    ArchiveExplorer,
    FieldStats,
    SavepointRef,
    _transition,
    component_labels,
    experiment_data_path,
    format_table,
    resolve_experiment,
    summarize,
)


# The '--only-group scripts' environment of './scripts/test' installs neither the icon4py
# model packages nor their dependencies, so the parts that read tracer names and experiment
# descriptions from them can only be exercised where './scripts/run' runs the CLI itself.
def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _requires(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not _installed(module),
        reason=f"'{module}' is not installed in the scripts test environment",
    )


requires_common = _requires("icon4py.model.common")
requires_driver = _requires("icon4py.model.standalone_driver")


def test_format_table_pads_columns():
    table = format_table(["a", "bbb"], [["xx", "y"]])
    assert table.splitlines() == ["a   bbb", "--  ---", "xx  y"]


def test_format_table_without_rows():
    assert format_table(["a"], []) == "(no rows)"


@requires_common
def test_component_labels_of_tracer_fields():
    assert component_labels("tracers", 6) == ("qv", "qc", "qi", "qr", "qs", "qg")
    assert component_labels("tracers_now", 2) == ("qv", "qc")
    assert component_labels("grf_tend_tracers", 6)[0] == "qv"


@requires_common
def test_component_labels_of_unknown_component_index():
    assert component_labels("tracers", 7)[6] == "idx6"


def test_component_labels_of_plain_field():
    assert component_labels("theta_v", 6) is None


def test_summarize_reports_range_and_mean():
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    stats = summarize(values, savepoint="sp", date="d", field="f", component=None)
    assert (stats.min, stats.max, stats.mean) == (0.0, 3.0, 1.5)
    assert stats.nonzero_fraction == 0.75
    assert stats.n_nonfinite == 0
    assert not stats.all_zero


def test_summarize_flags_all_zero_field():
    stats = summarize(np.zeros((3, 4)), savepoint="sp", date=None, field="f", component=None)
    assert stats.all_zero
    assert stats.nonzero_fraction == 0.0


def test_summarize_ignores_non_finite_values_in_the_range():
    values = np.array([1.0, np.nan, np.inf, 3.0])
    stats = summarize(values, savepoint="sp", date=None, field="f", component=None)
    assert (stats.min, stats.max) == (1.0, 3.0)
    assert stats.n_nonfinite == 2


def test_summarize_squeezes_singleton_axes():
    stats = summarize(np.zeros((5, 4, 1)), savepoint="sp", date=None, field="f", component=None)
    assert stats.shape == (5, 4)


def test_field_stats_full_name():
    def stats(component):
        return FieldStats("sp", None, "tracers", component, (1,), 0.0, 0.0, 0.0, 1.0, 0)

    assert stats("qv").full_name == "tracers.qv"
    assert stats(None).full_name == "tracers"


def test_transition_collapses_equal_endpoints():
    assert _transition("a", "b") == "a -> b"
    assert _transition("a", "a") == "a"
    assert _transition(None, None) is None


def test_savepoint_ref_label():
    assert SavepointRef(0, "sp", {"date": "d"}).label == "sp @ d"
    assert SavepointRef(3, "sp", {}).label == "sp #3"


@requires_driver
def test_resolve_experiment_by_icon_name_and_attribute():
    by_name = resolve_experiment("exclaim_ape_aesPhys")
    by_attribute = resolve_experiment("EXCLAIM_APE_AES")
    assert by_name is by_attribute
    assert by_name.name == "exclaim_ape_aesPhys"


@requires_driver
def test_resolve_experiment_rejects_unknown_name():
    with pytest.raises(typer.BadParameter, match="Unknown experiment"):
        resolve_experiment("not_an_experiment")


@requires_driver
def test_experiment_data_path_encodes_ranks_and_version():
    description = dataclasses.replace(resolve_experiment("exclaim_ape_aesPhys"), version=7)
    path = experiment_data_path(description, comm_size=4)
    assert path.parent.name == "mpitask4_exclaim_ape_aesPhys_v07"
    assert path.name == "ser_data"


def test_archive_explorer_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No serialized data directory"):
        ArchiveExplorer(tmp_path / "absent")

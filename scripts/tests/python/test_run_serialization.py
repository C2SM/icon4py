# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the serialization run driver."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
import run_serialization
import typer

from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs


@pytest.fixture(autouse=True)
def testing_definitions() -> None:
    # The driver imports 'icon4py.model.testing' lazily inside the command to keep CLI
    # startup fast, so the module globals have to be populated for direct calls.
    run_serialization.test_defs = test_defs
    run_serialization.dt_utils = dt_utils


@pytest.fixture
def settings(tmp_path: pathlib.Path) -> run_serialization.SerializationSettings:
    return dataclasses.replace(
        run_serialization.SerializationSettings.defaults(),
        icon4py_repo_dir=tmp_path,
        iconf90_repo_dir=tmp_path / "no-icon-here",
    )


def test_preflight_refuses_a_modified_checkout(settings, tmp_path: pathlib.Path) -> None:
    # Data generated from uncommitted changes cannot be reproduced later.
    run_serialization.run_command(["git", "init"], cwd=tmp_path)
    (tmp_path / "untracked.txt").write_text("x\n")

    with pytest.raises(typer.BadParameter, match="uncommitted changes"):
        run_serialization.preflight(settings=settings)


def test_preflight_without_an_icon_source_tree(settings, capsys) -> None:
    # The ICON tree only exists in the icon-exclaim layout; missing it is not an error.
    run_serialization.run_command(["git", "init"], cwd=settings.icon4py_repo_dir)

    run_serialization.preflight(settings=settings)

    assert "No ICON source tree" in capsys.readouterr().out


def test_failed_task_is_reported_not_raised() -> None:
    # One bad task must not discard the other 17 of a multi-hour campaign.
    broken = dataclasses.replace(
        run_serialization.SerializationSettings.defaults(),
        runscript_dir=pathlib.Path("/nonexistent"),
        build_dir=pathlib.Path("/nonexistent"),
    )

    result = run_serialization.run_experiment(test_defs.Experiments.GAUSS3D, 1, settings=broken)

    assert result.error is not None
    assert result.experiment_name == "exclaim_gauss3d"


def test_experiment_versions_are_explicit() -> None:
    # A shared default meant bumping one experiment invalidated the archives of all six.
    version = next(
        field
        for field in dataclasses.fields(test_defs.ExperimentDescription)
        if field.name == "version"
    )
    assert version.default is dataclasses.MISSING

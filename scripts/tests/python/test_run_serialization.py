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

from icon4py.model.testing import definitions as test_defs


@pytest.fixture(autouse=True)
def testing_definitions() -> None:
    # The driver imports 'icon4py.model.testing' lazily inside the command to keep CLI
    # startup fast, so the module global has to be populated for direct calls.
    run_serialization.test_defs = test_defs


def test_defaults_runs_every_experiment_and_comm_size() -> None:
    settings = run_serialization.SerializationSettings.defaults()

    assert len(settings.experiment_descriptions) == 6
    assert settings.comm_sizes == [1, 2, 4]


def test_defaults_selects_requested_experiments() -> None:
    settings = run_serialization.SerializationSettings.defaults(
        experiment_names=["exclaim_gauss3d"], comm_sizes=[4]
    )

    assert [e.name for e in settings.experiment_descriptions] == ["exclaim_gauss3d"]
    assert settings.comm_sizes == [4]


def test_defaults_rejects_unknown_experiment() -> None:
    with pytest.raises(typer.BadParameter, match="unknown experiments: nonsense"):
        run_serialization.SerializationSettings.defaults(experiment_names=["nonsense"])


def test_defaults_rejects_unsupported_comm_size() -> None:
    with pytest.raises(typer.BadParameter, match="unsupported communicator sizes"):
        run_serialization.SerializationSettings.defaults(comm_sizes=[3])


def test_experiment_versions_are_explicit() -> None:
    # A shared default would make bumping one experiment invalidate all the others.
    version_field = next(
        field
        for field in dataclasses.fields(test_defs.ExperimentDescription)
        if field.name == "version"
    )
    assert version_field.default is dataclasses.MISSING

    for experiment in run_serialization.SerializationSettings.defaults().experiment_descriptions:
        assert isinstance(experiment.version, int)


def test_task_result_is_json_serializable() -> None:
    result = run_serialization.TaskResult(
        experiment="exclaim_gauss3d",
        version=5,
        comm_size=1,
        status="ok",
        job_id="1234",
        tar_path=pathlib.Path("/tmp/mpitask1_exclaim_gauss3d_v05.tar.gz"),
    )

    assert result.as_dict()["tar_path"] == "/tmp/mpitask1_exclaim_gauss3d_v05.tar.gz"
    assert (
        run_serialization.TaskResult(
            experiment="x", version=5, comm_size=1, status="failed", error="boom"
        ).as_dict()["tar_path"]
        is None
    )

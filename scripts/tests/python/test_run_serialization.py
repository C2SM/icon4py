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
import shutil

import pytest
import run_serialization
import serdata
import typer

from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs


@pytest.fixture(autouse=True)
def testing_definitions() -> None:
    # The driver imports 'icon4py.model.testing' lazily inside the command to keep CLI
    # startup fast, so the module globals have to be populated for direct calls.
    run_serialization.test_defs = test_defs
    run_serialization.dt_utils = dt_utils


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


def test_archive_dirname_pattern_matches_the_authoritative_builder() -> None:
    # 'serdata.parse_archive_dirname' is the inverse of the name that
    # 'datatest_utils' builds, but cannot import it. Pin the two together here.
    for experiment in run_serialization.SerializationSettings.defaults().experiment_descriptions:
        for comm_size in (1, 2, 4):
            name = dt_utils.get_ranked_experiment_name_with_version(experiment, comm_size)
            assert serdata.parse_archive_dirname(name) == (
                comm_size,
                experiment.name,
                experiment.version,
            )


def test_comparison_without_a_baseline_is_unverified(tmp_path: pathlib.Path) -> None:
    # The first regeneration under this scheme has no predecessor on disk. It must say
    # so rather than report a clean comparison that never happened.
    settings = dataclasses.replace(
        run_serialization.SerializationSettings.defaults(), output_root=tmp_path
    )
    archive = tmp_path / "mpitask1_exclaim_gauss3d_v05"
    shutil.copytree(pathlib.Path(__file__).parent / "data" / "v05", archive)

    verdict, report_path = run_serialization.compare_with_previous_version(
        archive, test_defs.Experiments.GAUSS3D, 1, settings=settings
    )

    assert verdict == "UNVERIFIED"
    # the report is named after the archive, not after the tarball: '.tar.gz' has two
    # suffixes, so deriving the name from the tarball drops only the last one
    assert report_path == tmp_path / "reports" / "mpitask1_exclaim_gauss3d_v05.md"
    assert report_path.is_file()


def test_next_steps_point_at_the_reports_that_need_review(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    settings = dataclasses.replace(
        run_serialization.SerializationSettings.defaults(), output_root=tmp_path
    )
    result = run_serialization.TaskResult(
        experiment="exclaim_gauss3d",
        version=6,
        comm_size=1,
        status="ok",
        tar_path=tmp_path / "mpitask1_exclaim_gauss3d_v06.tar.gz",
        verdict="REVIEW",
        report_path=tmp_path / "reports" / "mpitask1_exclaim_gauss3d_v06.md",
    )

    run_serialization.print_next_steps([result], settings=settings)

    printed = capsys.readouterr().out
    assert "reports/mpitask1_exclaim_gauss3d_v06.md" in printed
    assert "exclaim_gauss3d: version=6" in printed

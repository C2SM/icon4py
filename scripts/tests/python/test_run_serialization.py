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
import typer

from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    serialized_data,
)


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
        experiment_name="exclaim_gauss3d",
        experiment_version=5,
        comm_size=1,
        status="ok",
        job_id="1234",
        tar_path=pathlib.Path("/tmp/mpitask1_exclaim_gauss3d_v05.tar.gz"),
    )

    assert result.as_dict()["tar_path"] == "/tmp/mpitask1_exclaim_gauss3d_v05.tar.gz"
    assert (
        run_serialization.TaskResult(
            experiment_name="x", experiment_version=5, comm_size=1, status="failed", error="boom"
        ).as_dict()["tar_path"]
        is None
    )


def test_archive_dirname_pattern_matches_the_authoritative_builder() -> None:
    # 'serialized_data.parse_archive_dirname' is the inverse of the name that
    # 'datatest_utils' builds, but cannot import it. Pin the two together here.
    for experiment in run_serialization.SerializationSettings.defaults().experiment_descriptions:
        for comm_size in (1, 2, 4):
            name = dt_utils.get_ranked_experiment_name_with_version(experiment, comm_size)
            assert serialized_data.parse_archive_dirname(name) == (
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
    # the archive fixtures live with the library that reads them
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    shutil.copytree(repo_root / "model/testing/tests/testing/unit_tests/data/v05", archive)

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
        experiment_name="exclaim_gauss3d",
        experiment_version=6,
        comm_size=1,
        status="ok",
        tar_path=tmp_path / "mpitask1_exclaim_gauss3d_v06.tar.gz",
        verdict="REVIEW",
        report_path=tmp_path / "reports" / "mpitask1_exclaim_gauss3d_v06.md",
    )

    run_serialization.print_next_steps([result], settings=settings)

    printed = capsys.readouterr().out
    assert "reports/mpitask1_exclaim_gauss3d_v06.md" in printed


def test_next_steps_are_printed_after_a_successful_campaign() -> None:
    # The helper was once defined and never called; a run must actually reach it.
    source = pathlib.Path(run_serialization.__file__).read_text()
    body = source.split("def run_serialization(", 1)[1]
    assert "print_next_steps(results, settings=settings)" in body


def test_next_steps_ignores_failed_tasks(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    settings = dataclasses.replace(
        run_serialization.SerializationSettings.defaults(), output_root=tmp_path
    )
    failed = run_serialization.TaskResult(
        experiment_name="exclaim_gauss3d",
        experiment_version=6,
        comm_size=1,
        status="failed",
        error="boom",
    )

    run_serialization.print_next_steps([failed], settings=settings)

    # a failed task has no verdict, report or tarball to point at
    assert "None" not in capsys.readouterr().out


def test_every_serdata_command_is_registered_before_the_entry_point() -> None:
    # Commands defined after the '__main__' guard are missing when the file is run
    # directly through its shebang.
    source = (pathlib.Path(run_serialization.__file__).parent / "serdata.py").read_text()
    guard = source.index('if __name__ == "__main__":')
    assert "@cli.command()" not in source[guard:]


def test_run_tests_flag_reaches_the_datatest_sweep() -> None:
    # The flag was once accepted and wired to nothing.
    body = (
        pathlib.Path(run_serialization.__file__).read_text().split("def run_serialization(", 1)[1]
    )
    assert "run_datatests(settings=settings)" in body

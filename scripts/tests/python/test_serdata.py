# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the serialized-data metadata and drift-detection helpers."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import subprocess

import pytest
from serdata import (
    BannerParseError,
    backfill_archive_metadata,
    classify_savepoint,
    compare_archives,
    diff_fingerprints,
    diff_namelists,
    fingerprint_archive,
    harvest_git,
    parse_archive_dirname,
    parse_icon_log_banner,
    read_icon_banner_from_archive,
    read_namelist,
    render_diff_report,
    render_value_change,
    summarise_differences,
    unclassified_savepoints,
)


DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def banner_log() -> str:
    return (DATA_DIR / "LOG.banner_sample.o").read_text()


def test_parse_banner_extracts_icon_revision(banner_log: str) -> None:
    banner = parse_icon_log_banner(banner_log)

    assert banner["sha"] == "97986bc3592cc05c799717c70345f27a8c275d8d"
    assert (
        banner["describe"] == "icon-2025.10-dwd-2.0-242-g97986bc3592cc05c799717c70345f27a8c275d8d"
    )
    assert banner["repository"] == "git@gitlab.dkrz.de:icon/icon-nwp.git"
    assert banner["branch"] == "serialize_tmx"
    assert banner["version"] == "2026.04"


def test_parse_banner_extracts_build_identity(banner_log: str) -> None:
    banner = parse_icon_log_banner(banner_log)

    assert banner["build_date"] == "20260703"
    assert banner["build_time"] == "115609"
    assert banner["host"] == "nid005237"
    assert banner["executable"].endswith("/build_serialize/bin/icon")


def test_parse_banner_collects_externals_but_not_toolchain(banner_log: str) -> None:
    banner = parse_icon_log_banner(banner_log)

    externals = banner["externals"]
    assert externals["icon-land"] == "icon-land-2026.04-6-ga7583e49db9c7286f1e1377a7c8409d0c11425c5"
    assert externals["dace"] == "icon-nwp-2001-0-ga56af966bfc607d64b96580d1f35d70996800547"
    assert externals["ecrad"].startswith("ecrad-safeguard-09666303-13-g")
    assert externals["mtime"] == "1.3.0-1-g04f02ccbd765104a1570355c5d3fdfcfdad11c4c"
    # nested entries expose a 'revision' line one level deeper than their name
    assert externals["cdi"] == "cdi-2.6.0-1-g60e0a3b9899c8021437686d720702a978a5bb63d"
    # toolchain lives in its own section, not mixed into externals
    assert "eccodes" not in externals
    assert "fortran" not in externals


def test_parse_banner_collects_toolchain(banner_log: str) -> None:
    toolchain = parse_icon_log_banner(banner_log)["toolchain"]

    assert toolchain["eccodes"] == "2.36.4"
    assert toolchain["netcdf-c"] == "4.9.2"
    assert toolchain["fortran"] == "NVHPC 25.1.0 (nvfortran 25.1-0)"
    assert toolchain["mpi"].startswith("MPI VERSION")


def test_parse_banner_stops_before_the_model_log(banner_log: str) -> None:
    banner = parse_icon_log_banner(banner_log)

    assert "master_control" not in banner["externals"]
    assert "master_control" not in banner["toolchain"]


def test_parse_banner_raises_when_absent() -> None:
    with pytest.raises(BannerParseError, match="No ICON version banner"):
        parse_icon_log_banner("srun: job 1 queued\nsome unrelated output\n")


def test_parse_banner_raises_when_revision_missing(banner_log: str) -> None:
    mutilated = "\n".join(
        line for line in banner_log.splitlines() if not line.startswith(" revision:")
    )
    with pytest.raises(BannerParseError, match="revision"):
        parse_icon_log_banner(mutilated)


def test_read_banner_from_archive(tmp_path: pathlib.Path, banner_log: str) -> None:
    (tmp_path / "LOG.exp.foo_sb.run.12345.o").write_text(banner_log)

    banner = read_icon_banner_from_archive(tmp_path)

    assert banner["sha"] == "97986bc3592cc05c799717c70345f27a8c275d8d"
    assert banner["source"] == "LOG.exp.foo_sb.run.12345.o"


def test_read_banner_from_archive_without_log(tmp_path: pathlib.Path) -> None:
    with pytest.raises(BannerParseError, match="No ICON log"):
        read_icon_banner_from_archive(tmp_path)


def test_read_banner_from_archive_with_ambiguous_logs(
    tmp_path: pathlib.Path, banner_log: str
) -> None:
    (tmp_path / "LOG.exp.foo_sb.run.1.o").write_text(banner_log)
    (tmp_path / "LOG.exp.foo_sb.run.2.o").write_text(banner_log)

    with pytest.raises(BannerParseError, match="several ICON logs"):
        read_icon_banner_from_archive(tmp_path)


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "some_branch")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "file.txt").write_text("content\n")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_harvest_git_reports_clean_tree(git_repo: pathlib.Path) -> None:
    state = harvest_git(git_repo)

    assert len(state["sha"]) == 40
    assert state["branch"] == "some_branch"
    assert state["dirty"] is False


def test_harvest_git_reports_dirty_tree(git_repo: pathlib.Path) -> None:
    (git_repo / "file.txt").write_text("modified\n")

    assert harvest_git(git_repo)["dirty"] is True


def test_harvest_git_on_non_repository(tmp_path: pathlib.Path) -> None:
    assert harvest_git(tmp_path) == {}


# --- savepoint classification -------------------------------------------------


def test_classify_static_savepoints() -> None:
    assert classify_savepoint("icon-grid", {"nlev": 35}) == "static"
    assert classify_savepoint("metric-state", {"id": 1}) == "static"
    assert classify_savepoint("tmx-init", {"id": 1}) == "static"


def test_classify_initial_state_wins_over_static() -> None:
    # 'diagnostics' and 'prognostics' carry neither a date nor a dyn_timestep, so a
    # "written once" heuristic alone would swallow them into the static class.
    meta = {"id": 1, "location": "initial-state"}
    assert classify_savepoint("diagnostics", meta) == "initial-state"
    assert classify_savepoint("prognostics", meta) == "initial-state"


def test_classify_evolving_savepoints() -> None:
    assert classify_savepoint("diffusion-init", {"date": "...", "id": 1}) == "evolving"
    # named like an initial state, but written with a date: not the t=0 dump
    assert classify_savepoint("jabw-initial-state-exit", {"date": "...", "id": 1}) == "evolving"


def test_unclassified_savepoints_are_flagged() -> None:
    savepoints = [
        ("icon-grid", {"id": 1}),
        ("diffusion-init", {"date": "...", "id": 1}),
        ("brand-new-startup-savepoint", {"id": 1}),
    ]

    # looks time-invariant but is not in the explicit list: a human has to classify it
    assert unclassified_savepoints(savepoints) == ["brand-new-startup-savepoint"]


# --- fingerprinting -----------------------------------------------------------


def test_fingerprint_covers_every_rank() -> None:
    fingerprint = fingerprint_archive(DATA_DIR / "v05")

    assert fingerprint.ranks == 2
    assert len(fingerprint.records) == 86  # 43 records on each of the two ranks
    assert {rank for rank, *_ in fingerprint.records} == {0, 1}


def test_fingerprint_keys_are_savepoint_ordinal_and_field() -> None:
    fingerprint = fingerprint_archive(DATA_DIR / "v05")

    diffusion = sorted(
        key for key in fingerprint.records if key[1] == "diffusion-init" and key[0] == 0
    )
    # 'diffusion-init' is written three times; the ordinal distinguishes the occurrences
    assert {key[2] for key in diffusion} == {0, 1, 2}


def test_fingerprint_resolves_occurrences_by_list_position() -> None:
    # 'fields_table[field][i][0]' is a byte offset, not an occurrence id. Keying on it
    # would collapse the occurrences of a repeatedly written field onto one digest.
    fingerprint = fingerprint_archive(DATA_DIR / "v05")

    tracers = {
        key[2]: digest
        for key, digest in fingerprint.records.items()
        if key[0] == 0 and key[3] == "tracers"
    }
    assert len(tracers) == 3
    assert len(set(tracers.values())) == 3


def test_fingerprint_keeps_digests_verbatim() -> None:
    # Serialbox strips leading zeros per byte, so digests are 54-64 characters and
    # cannot be padded back to a fixed width without corrupting the alignment.
    digests = fingerprint_archive(DATA_DIR / "v05").records.values()

    assert any(len(digest) < 64 for digest in digests)
    assert all(54 <= len(digest) <= 64 for digest in digests)


def test_fingerprint_distinguishes_ranks() -> None:
    fingerprint = fingerprint_archive(DATA_DIR / "v05")

    differing = [
        key
        for key in fingerprint.records
        if key[0] == 0 and fingerprint.records[key] != fingerprint.records[(1, *key[1:])]
    ]
    # a decomposed domain gives different data per rank, so ranks cannot be merged
    assert differing


def test_fingerprint_ignores_declared_but_unwritten_fields() -> None:
    # 'field_map' is a superset of 'fields_table'; iterating it would raise a KeyError
    fingerprint = fingerprint_archive(DATA_DIR / "v05")

    assert not [key for key in fingerprint.records if key[3].startswith("tracers_")]


def test_fingerprint_records_savepoint_classes() -> None:
    fingerprint = fingerprint_archive(DATA_DIR / "v06")

    assert fingerprint.classes["icon-grid"] == "static"
    assert fingerprint.classes["tmx-init"] == "static"
    assert fingerprint.classes["prognostics"] == "initial-state"
    assert fingerprint.classes["advection-exit"] == "evolving"


# --- diffing ------------------------------------------------------------------


def test_diff_of_a_real_regeneration() -> None:
    # The v05 -> v06 regeneration of this experiment added the TMX turbulence scheme.
    # It must show up as new savepoints and an evolving trajectory, with the grid and
    # the initial state untouched.
    diff = diff_fingerprints(
        fingerprint_archive(DATA_DIR / "v05"), fingerprint_archive(DATA_DIR / "v06")
    )

    assert diff.per_class["static"].changed == []
    assert diff.per_class["initial-state"].changed == []
    assert diff.per_class["initial-state"].added == []
    assert diff.savepoints_added == ["tmx-init"]
    assert diff.savepoints_removed == []
    # tmx-init contributes new static records, and the report can name them
    assert {key[1] for key in diff.per_class["static"].added} == {"tmx-init"}


def test_diff_reports_changed_records() -> None:
    old = fingerprint_archive(DATA_DIR / "v05")
    tampered = fingerprint_archive(DATA_DIR / "v05")
    key = next(key for key in tampered.records if key[1] == "icon-grid")
    tampered.records[key] = "DEADBEEF"

    diff = diff_fingerprints(old, tampered)

    assert diff.per_class["static"].changed == [key]
    assert diff.per_class["evolving"].changed == []


def test_fingerprint_flags_savepoints_nobody_classified() -> None:
    # The fixtures contain only savepoints that are already classified.
    assert fingerprint_archive(DATA_DIR / "v06").unclassified == []


# --- namelists ----------------------------------------------------------------


def test_read_namelist_of_an_archive() -> None:
    namelist = read_namelist(DATA_DIR / "v05")

    assert namelist["grid_nml"]["nroot"] == 2


def test_read_namelist_when_absent(tmp_path: pathlib.Path) -> None:
    assert read_namelist(tmp_path) == {}


def test_diff_namelists_reports_changed_defaults() -> None:
    # An upstream change to a namelist default shows up here and nowhere else: the
    # input namelists only carry the values the experiment sets explicitly.
    diff = diff_namelists(read_namelist(DATA_DIR / "v05"), read_namelist(DATA_DIR / "v06"))

    assert diff.changed == [("nonhydrostatic_nml", "damp_height", 45000.0, 12500.0)]
    assert diff.added == [("nonhydrostatic_nml", "new_key")]
    assert diff.removed == [("nonhydrostatic_nml", "old_key")]


def test_diff_namelists_of_identical_input() -> None:
    namelist = read_namelist(DATA_DIR / "v05")

    diff = diff_namelists(namelist, namelist)

    assert (diff.changed, diff.added, diff.removed) == ([], [], [])


# --- comparison and report ----------------------------------------------------


def test_compare_archives_verdict_is_ok_when_only_evolving_changed() -> None:
    comparison = compare_archives(DATA_DIR / "v05", DATA_DIR / "v05")

    assert comparison.verdict == "OK"


def test_compare_archives_flags_a_changed_static_record() -> None:
    comparison = compare_archives(DATA_DIR / "v05", DATA_DIR / "v06")
    tampered = dataclasses.replace(
        comparison,
        fingerprints=dataclasses.replace(
            comparison.fingerprints,
            per_class={
                **comparison.fingerprints.per_class,
                "static": dataclasses.replace(
                    comparison.fingerprints.per_class["static"],
                    changed=[(0, "icon-grid", 0, "primal_normal_cell_x")],
                ),
            },
        ),
    )

    assert tampered.verdict == "REVIEW"


def test_report_names_changed_static_fields_and_counts_the_rest() -> None:
    comparison = compare_archives(DATA_DIR / "v05", DATA_DIR / "v06")

    report = render_diff_report(comparison)

    assert "VERDICT: OK" in report
    assert "tmx-init" in report  # a savepoint the regeneration added
    assert "damp_height" in report  # the namelist default that drifted
    assert "icon 97986bc359" in report  # provenance, abbreviated


def test_report_states_when_there_is_no_baseline(tmp_path: pathlib.Path) -> None:
    comparison = compare_archives(None, DATA_DIR / "v05")

    report = render_diff_report(comparison)

    assert "BASELINE: none" in report


# --- archive directory names and backfill -------------------------------------


def test_parse_archive_dirname() -> None:
    assert parse_archive_dirname("mpitask4_exclaim_ape_aesPhys_v06") == (
        4,
        "exclaim_ape_aesPhys",
        6,
    )


def test_parse_archive_dirname_rejects_other_directories() -> None:
    assert parse_archive_dirname("grids") is None
    assert parse_archive_dirname("mpitask4_exclaim_ape_aesPhys") is None


def test_backfill_writes_metadata_for_existing_archives(tmp_path: pathlib.Path) -> None:
    # Archives generated before the metadata file existed still carry the ICON log, so
    # their provenance can be recovered locally without regenerating anything.
    archive = tmp_path / "mpitask2_exclaim_ape_aesPhys_v05"
    shutil.copytree(DATA_DIR / "v05", archive)

    written = backfill_archive_metadata(tmp_path)

    assert written == [archive / "archive_metadata.json"]
    metadata = json.loads(written[0].read_text())
    assert metadata["archive"] == {
        "experiment": "exclaim_ape_aesPhys",
        "version": 5,
        "comm_size": 2,
        "backfilled": True,
    }
    assert metadata["provenance"]["icon"]["sha"] == "97986bc3592cc05c799717c70345f27a8c275d8d"


def test_backfill_skips_archives_that_already_have_metadata(tmp_path: pathlib.Path) -> None:
    archive = tmp_path / "mpitask1_exclaim_gauss3d_v05"
    shutil.copytree(DATA_DIR / "v05", archive)
    (archive / "archive_metadata.json").write_text("{}")

    assert backfill_archive_metadata(tmp_path) == []


def test_backfill_reports_archives_it_cannot_read(tmp_path: pathlib.Path) -> None:
    broken = tmp_path / "mpitask1_exclaim_gauss3d_v05"
    broken.mkdir()

    # one unreadable archive must not stop the others
    assert backfill_archive_metadata(tmp_path) == []


def test_report_points_at_the_differing_element_of_an_array_value() -> None:
    # Namelist array values are long and mostly identical; truncating both sides makes
    # the two look the same, which is worse than useless.
    old = {"aes_vdf_nml": {"config": [1.0, 0.17, 0.25, 0.0]}}
    new = {"aes_vdf_nml": {"config": [1.0, 0.42, 0.25, 0.0]}}

    rendered = render_value_change(*diff_namelists(old, new).changed[0][2:])

    assert rendered == "[1] 0.17 -> 0.42"


def test_report_falls_back_to_whole_values_for_scalars() -> None:
    assert render_value_change(45000.0, 12500.0) == "45000.0 -> 12500.0"


def test_report_strips_fortran_string_padding() -> None:
    # A value that Fortran padded to its declared width must not read as "unchanged
    # whitespace"; quoting is what makes an emptied string visible.
    assert render_value_change("                ", "PT300S     ") == "'' -> 'PT300S'"


# --- triaging a changed record ------------------------------------------------


def test_summarise_differences_reports_positions_and_values() -> None:
    old = [[0.0, 1.0], [2.0, 3.0]]
    new = [[0.0, 9.0], [2.0, 3.0]]

    summary = summarise_differences(old, new, limit=5)

    assert summary.count == 1
    assert summary.total == 4
    assert summary.samples == [((0, 1), 1.0, 9.0)]


def test_summarise_differences_caps_the_samples() -> None:
    old = [[0.0] * 10]
    new = [[float(i + 1) for i in range(10)]]

    summary = summarise_differences(old, new, limit=3)

    assert summary.count == 10
    assert len(summary.samples) == 3


def test_summarise_differences_of_identical_arrays() -> None:
    summary = summarise_differences([[1.0, 2.0]], [[1.0, 2.0]], limit=5)

    assert summary.count == 0
    assert summary.samples == []

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reading the ICON revision out of an archive."""

from __future__ import annotations

import pathlib

import pytest

from icon4py.model.testing import provenance


REVISION = "icon-2026.04-dwd-1.9-136-g1df503335b8726dc445659c21020f4f09d9acc3d"

# The shape ICON's startup banner has inside the slurm log, with enough of the
# surrounding output to show that the reader has to find the line rather than index it.
LOG = f"""+ srun -n 1 /capstor/store/.../build_serialize/bin/icon

 executable: /capstor/store/.../build_serialize/bin/icon
 date: 20260727
 time: 131015
 host: nid005281 (Linux 6.4.0 aarch64)
 version: 2026.04
 revision: {REVISION}
 repository: git@gitlab.dkrz.de:icon/icon-nwp.git
 local branch: serialize_tmx
 model components:
   ICON-Land:
     revision: icon-land-2026.04-18-geb24d7431a16ef66bae693a7ef95c32ca47862ff
 master_control: start model initialization.
"""


@pytest.fixture
def archive(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "LOG.exp.foo_sb.run.986945.o").write_text(LOG)
    return tmp_path


def test_read_icon_revision(archive: pathlib.Path) -> None:
    # The whole describe string, not the bare sha: it is a valid git revision and its
    # release lineage is what distinguishes a small bump from a jump to another release.
    assert provenance.read_icon_revision(archive) == REVISION


def test_read_icon_revision_ignores_nested_components(archive: pathlib.Path) -> None:
    # ICON-Land also announces a 'revision:', indented further.
    assert not provenance.read_icon_revision(archive).startswith("icon-land")


def test_read_icon_revision_without_a_log(tmp_path: pathlib.Path) -> None:
    assert provenance.read_icon_revision(tmp_path) is None


def test_recording_an_archive_without_a_log_does_not_raise(tmp_path: pathlib.Path) -> None:
    # A missing or unreadable log must never fail a test session over bookkeeping.
    provenance.record("mpitask1_nothing_here_v01", tmp_path)

    assert provenance.seen()["mpitask1_nothing_here_v01"] == "unknown"


def test_merge_takes_in_what_another_process_saw() -> None:
    # Under xdist the archives are downloaded in the workers, but the summary is
    # rendered on the controller, whose module state is otherwise never touched.
    provenance.merge({"mpitask1_exclaim_gauss3d_v05": REVISION})

    assert provenance.seen()["mpitask1_exclaim_gauss3d_v05"] == REVISION

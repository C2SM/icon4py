# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
from click.testing import CliRunner

import icon4pytools.liskov.parsing.parse as ts
from icon4pytools.liskov.parsing.scan import DirectivesScanner


@pytest.fixture
def make_f90_tmpfile(tmp_path) -> Path:
    """Fixture factory which creates a temporary Fortran file.

    Args:
        content: Content to be present in the file.
    """

    def _make_f90_tmpfile(content: str):
        fn = tmp_path / "tmp.f90"
        with open(fn, "w") as f:
            f.write(content)
        return fn

    return _make_f90_tmpfile


@pytest.fixture
def cli():
    return CliRunner()


def scan_for_directives(fpath: Path) -> list[ts.RawDirective]:
    collector = DirectivesScanner(fpath)
    return collector()


def insert_new_lines(fname: Path, lines: list[str]) -> None:
    """Append new lines into file."""
    with open(fname, "a") as f:
        for ln in lines:
            f.write(f"{ln}\n")

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import pytest
from click.testing import CliRunner

from icon4py.liskov.parsing.scan import DirectivesScanner
from icon4py.liskov.parsing.types import RawDirective


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


def scan_for_directives(fpath: Path) -> list[RawDirective]:
    collector = DirectivesScanner(fpath)
    return collector.directives


def insert_new_lines(fname: Path, lines: list[str]):
    with open(fname, "a") as f:
        for ln in lines:
            f.write(f"{ln}\n")

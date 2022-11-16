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


@pytest.fixture
def make_f90_tmpfile(tmp_path_factory) -> Path:
    """Fixture factory which creates a temporary Fortran file.

    Args:
        content: Content to be present in the file.
    """

    def _make_f90_tmpfile(content: str):
        fn = tmp_path_factory.mktemp("testfiles") / "tmp.f90"
        fn.write_text(content)
        return fn

    return _make_f90_tmpfile


@pytest.fixture
def cli():
    return CliRunner()

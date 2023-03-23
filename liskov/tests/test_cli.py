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

import pytest
from samples.fortran_samples import (
    CONSECUTIVE_STENCIL,
    FREE_FORM_STENCIL,
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_STENCIL,
)

from icon4py.liskov.cli import main


@pytest.fixture
def outfile(tmp_path):
    return str(tmp_path / "gen.f90")


@pytest.mark.parametrize("file", [NO_DIRECTIVES_STENCIL])
def test_cli_no_directives(make_f90_tmpfile, cli, file, outfile):
    fpath = str(make_f90_tmpfile(content=file))
    result = cli.invoke(main, [fpath, outfile])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "file, profile",
    [
        (NO_DIRECTIVES_STENCIL, False),
        (SINGLE_STENCIL, False),
        (CONSECUTIVE_STENCIL, False),
        (FREE_FORM_STENCIL, False),
        (MULTIPLE_STENCILS, False),
        (SINGLE_STENCIL, True),
        (CONSECUTIVE_STENCIL, True),
        (FREE_FORM_STENCIL, True),
        (MULTIPLE_STENCILS, True),
    ],
)
def test_cli(make_f90_tmpfile, cli, file, outfile, profile):
    fpath = str(make_f90_tmpfile(content=file))
    args = [fpath, outfile]
    if profile:
        args.append("--profile")
    result = cli.invoke(main, args)
    assert result.exit_code == 0

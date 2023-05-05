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

from icon4py.liskov.cli import main
from icon4py.testutils.liskov_fortran_samples import (
    CONSECUTIVE_STENCIL,
    FREE_FORM_STENCIL,
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    REPEATED_STENCILS,
    SINGLE_STENCIL,
)


@pytest.fixture
def outfile(tmp_path):
    return str(tmp_path / "gen.f90")


@pytest.mark.parametrize(
    "file, options",
    [
        (NO_DIRECTIVES_STENCIL, ["--ppser"]),
        (NO_DIRECTIVES_STENCIL, []),
        (SINGLE_STENCIL, ["--ppser"]),
        (SINGLE_STENCIL, []),
        (CONSECUTIVE_STENCIL, ["--ppser"]),
        (CONSECUTIVE_STENCIL, []),
        (FREE_FORM_STENCIL, ["--ppser"]),
        (FREE_FORM_STENCIL, []),
        (MULTIPLE_STENCILS, ["--ppser"]),
        (MULTIPLE_STENCILS, []),
        (SINGLE_STENCIL, ["--ppser"]),
        (SINGLE_STENCIL, ["--profile"]),
        (CONSECUTIVE_STENCIL, ["--ppser", "--profile"]),
        (CONSECUTIVE_STENCIL, ["--profile"]),
        (FREE_FORM_STENCIL, ["--ppser", "--profile"]),
        (FREE_FORM_STENCIL, ["--profile"]),
        (MULTIPLE_STENCILS, ["--ppser", "--profile"]),
        (MULTIPLE_STENCILS, ["--profile"]),
        (REPEATED_STENCILS, ["--ppser", "--profile"]),
        (REPEATED_STENCILS, ["--profile"]),
    ],
)
def test_cli(make_f90_tmpfile, cli, file, outfile, options):
    fpath = str(make_f90_tmpfile(content=file))
    args = [fpath, outfile, *options, "-m"]
    result = cli.invoke(main, args)
    assert result.exit_code == 0


# todo: add test for wrong arguments

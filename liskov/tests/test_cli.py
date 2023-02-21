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


@pytest.mark.parametrize("file", [NO_DIRECTIVES_STENCIL])
def test_cli_no_directives(make_f90_tmpfile, cli, file):
    fpath = str(make_f90_tmpfile(content=file))
    result = cli.invoke(main, [fpath])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "file", [SINGLE_STENCIL, CONSECUTIVE_STENCIL, FREE_FORM_STENCIL, MULTIPLE_STENCILS]
)
def test_cli(make_f90_tmpfile, cli, file):
    fpath = str(make_f90_tmpfile(content=file))
    result = cli.invoke(main, [fpath])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "file", [SINGLE_STENCIL, CONSECUTIVE_STENCIL, FREE_FORM_STENCIL, MULTIPLE_STENCILS]
)
def test_cli_profile(make_f90_tmpfile, cli, file):
    fpath = str(make_f90_tmpfile(content=file))
    result = cli.invoke(main, [fpath, "--profile"])
    assert result.exit_code == 0

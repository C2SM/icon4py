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
from click.testing import CliRunner

from icon4py.f2ser.cli import main
from icon4py.f2ser.exceptions import MissingDerivedTypeError

@pytest.fixture
def outfile(tmp_path):
    return str(tmp_path / "gen.f90")


@pytest.fixture
def cli():
    return CliRunner()


def test_cli(diffusion_granule, diffusion_granule_deps, outfile, cli):
    inp = str(diffusion_granule)
    deps = [str(p) for p in diffusion_granule_deps]
    args = [inp, outfile, "-d", ",".join(deps)]
    result = cli.invoke(main, args)
    assert result.exit_code == 0

def test_cli_no_deps(no_deps_source_file, outfile, cli):
    inp = str(no_deps_source_file)
    args = [inp, outfile]
    result = cli.invoke(main, args)
    assert result.exit_code == 0

def test_cli_missing_deps(diffusion_granule, outfile, cli):
    inp = str(diffusion_granule)
    args = [inp, outfile]
    result = cli.invoke(main, args)
    assert result.exception == MissingDerivedTypeError

def test_cli_missing_source(not_existing_diffusion_granule, outfile, cli):
    inp = str(not_existing_diffusion_granule)
    args = [inp, outfile]
    result = cli.invoke(main, args)
    error_search = result.stdout.find("Invalid value for \'GRANULE_PATH\'")
    assert error_search != -1

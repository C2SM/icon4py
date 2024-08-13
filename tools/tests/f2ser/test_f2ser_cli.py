# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from click.testing import CliRunner

from icon4pytools.f2ser.cli import main
from icon4pytools.f2ser.exceptions import MissingDerivedTypeError


@pytest.fixture
def outfile(tmp_path):
    return str(tmp_path / "gen.f90")


@pytest.fixture
def cli():
    return CliRunner()


@pytest.mark.parametrize("multinode", [False, True])
def test_cli(diffusion_granule, diffusion_granule_deps, outfile, cli, multinode):
    inp = str(diffusion_granule)
    deps = [str(p) for p in diffusion_granule_deps]
    args = [inp, outfile, "-d", ",".join(deps)]
    if multinode:
        args.append("--multinode")
    result = cli.invoke(main, args)
    assert result.exit_code == 0


def test_cli_no_deps(no_deps_source_file, outfile, cli):
    inp = str(no_deps_source_file)
    args = [inp, outfile]
    result = cli.invoke(main, args)
    assert result.exit_code == 0


def test_cli_wrong_deps(diffusion_granule, samples_path, outfile, cli):
    inp = str(diffusion_granule)
    deps = [str(samples_path / "wrong_derived_types_example.f90")]
    args = [inp, outfile, "-d", *deps]
    result = cli.invoke(main, args)
    assert result.exit_code == 2
    assert "Invalid value for '--dependencies' / '-d'" in result.output


def test_cli_missing_deps(diffusion_granule, outfile, cli):
    inp = str(diffusion_granule)
    args = [inp, outfile]
    result = cli.invoke(main, args)
    assert isinstance(result.exception, MissingDerivedTypeError)


def test_cli_wrong_source(outfile, cli):
    inp = str("foo.90")
    args = [inp, outfile]
    result = cli.invoke(main, args)
    assert "Invalid value for 'GRANULE_PATH'" in result.output


def test_cli_missing_source(not_existing_diffusion_granule, outfile, cli):
    inp = str(not_existing_diffusion_granule)
    args = [inp, outfile]
    result = cli.invoke(main, args)
    assert isinstance(result.exception, SystemExit)
    assert "Invalid value for 'GRANULE_PATH'" in result.output

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

import itertools

import pytest

from icon4pytools.liskov.cli import main
from icon4pytools.liskov.external.exceptions import MissingCommandError

from .fortran_samples import (
    CONSECUTIVE_STENCIL,
    FREE_FORM_STENCIL,
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_STENCIL,
)


@pytest.fixture
def outfile(tmp_path):
    return str(tmp_path / "gen.f90")


test_cases = []

files = [
    ("NO_DIRECTIVES", NO_DIRECTIVES_STENCIL),
    ("SINGLE", SINGLE_STENCIL),
    ("CONSECUTIVE", CONSECUTIVE_STENCIL),
    ("FREE_FORM", FREE_FORM_STENCIL),
    ("MULTIPLE", MULTIPLE_STENCILS),
]

flags = {"serialise": ["--multinode"], "integrate": ["-p", "-m"]}

for file_name, file_content in files:
    for cmd in flags.keys():
        flag_combinations = []
        for r in range(1, len(flags[cmd]) + 1):
            flag_combinations.extend(itertools.combinations(flags[cmd], r))
        for flags_selected in flag_combinations:
            args = (file_name, file_content, cmd, list(flags_selected))
            test_cases.append(args)


@pytest.mark.parametrize(
    "file_name, file_content, cmd, cmd_flags",
    test_cases,
    ids=[
        "file={}, command={}, flags={}".format(file_name, cmd, ",".join(cmd_flags))
        for file_name, file_content, cmd, cmd_flags in test_cases
    ],
)
def test_cli(make_f90_tmpfile, cli, outfile, file_name, file_content, cmd, cmd_flags):
    fpath = str(make_f90_tmpfile(content=file_content))
    args = [cmd, *cmd_flags, fpath, outfile]
    result = cli.invoke(main, args)
    assert result.exit_code == 0


def test_cli_missing_command(cli):
    args = []
    result = cli.invoke(main, args)
    assert result.exit_code == 1
    assert isinstance(result.exception, MissingCommandError)

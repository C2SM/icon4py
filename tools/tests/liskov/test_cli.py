# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import itertools

import pytest

from icon4pytools.liskov.cli import main
from icon4pytools.liskov.external.exceptions import MissingCommandError

from .fortran_samples import (
    CONSECUTIVE_STENCIL,
    FREE_FORM_STENCIL,
    MULTIPLE_FUSED,
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_FUSED,
    SINGLE_STENCIL,
    SINGLE_STENCIL_WITH_COMMENTS,
)


@pytest.fixture
def outfile(tmp_path):
    return str(tmp_path / "gen.f90")


test_cases = []

files = [
    ("NO_DIRECTIVES", NO_DIRECTIVES_STENCIL),
    ("SINGLE", SINGLE_STENCIL),
    ("COMMENTS", SINGLE_STENCIL_WITH_COMMENTS),
    ("CONSECUTIVE", CONSECUTIVE_STENCIL),
    ("FREE_FORM", FREE_FORM_STENCIL),
    ("MULTIPLE", MULTIPLE_STENCILS),
    ("SINGLE_FUSED", SINGLE_FUSED),
    ("MULTIPLE_FUSED", MULTIPLE_FUSED),
]

flags = {"serialise": ["--multinode"], "integrate": ["-p", "-m", "-f", "-u"]}

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

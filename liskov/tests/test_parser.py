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

from icon4py.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    ParsingException,
)
from icon4py.liskov.parsing.parse import DirectivesParser
from icon4py.liskov.parsing.scan import DirectivesScanner
from icon4py.liskov.parsing.types import NoDirectivesFound, RawDirective
from icon4py.testutils.fortran_samples import (
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_STENCIL,
)


# todo: change naming of helper function to reflect scanning
def collect_directives(fpath: Path) -> list[RawDirective]:
    collector = DirectivesScanner(fpath)
    return collector.directives


# todo: move this to testutils?
def insert_new_lines(fname: Path, lines: list[str]):
    with open(fname, "a") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def test_directive_parser_single_stencil(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    directives = collect_directives(fpath)
    parser = DirectivesParser(directives)
    parsed = parser.parsed_directives

    directives = parsed["directives"]
    content = parsed["content"]

    # todo: check that each element is the expected one.
    # todo: combine stencil cases with expected elements using pytest parametrise
    assert len(directives) == 5
    assert len(content) == 5


def test_directive_parser_multiple_stencils(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    directives = collect_directives(fpath)
    parser = DirectivesParser(directives)
    parsed = parser.parsed_directives

    directives = parsed["directives"]
    content = parsed["content"]

    # todo: same as above
    assert len(directives) == 9
    assert len(content) == 5


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL FOO_DIRECTIVE()",
        "!$DSL BAR_DIRECTIVE()",
    ],
)
def test_directive_parser_parsing_exception(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])
    directives = collect_directives(fpath)

    with pytest.raises(ParsingException):
        DirectivesParser(directives)
        # todo: test for more specific exception
        # todo: check for specific error message
        # todo: introduce more stencil cases


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL START(stencil1, stencil2)",
        "!$DSL DECLARE(somefield; another_field)",
        "!$DSL IMPORT(field)",
    ],
)
def test_directive_parser_invalid_directive_syntax(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])
    directives = collect_directives(fpath)

    with pytest.raises(DirectiveSyntaxError):
        DirectivesParser(directives)
        # todo: introduce more stencil cases


def test_directive_parser_no_directives_found(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=NO_DIRECTIVES_STENCIL)
    directives = collect_directives(fpath)
    parser = DirectivesParser(directives)
    parsed = parser.parsed_directives
    assert isinstance(parsed, NoDirectivesFound)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL IMPORT()",
        "!$DSL END(name=mo_nh_diffusion_stencil_06)",
    ],
)
def test_directive_parser_repeated_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    insert_new_lines(fpath, [directive])
    directives = collect_directives(fpath)

    with pytest.raises(ParsingException):
        DirectivesParser(directives)
        # todo: introduce more stencil cases
        # todo: exception should be more specific
        # todo: check for error message.


@pytest.mark.parametrize(
    "directive",
    [
        """!$DSL IMPORT()""",
        """!$DSL END(name=mo_nh_diffusion_stencil_06)""",
    ],
)
def test_directive_parser_required_directives(make_f90_tmpfile, directive):
    new = SINGLE_STENCIL.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    directives = collect_directives(fpath)

    with pytest.raises(ParsingException):
        DirectivesParser(directives)
        # todo: introduce more stencil cases
        # todo: exception should be more specific
        # todo: check for error message.


@pytest.mark.parametrize(
    "directive",
    [
        """!$DSL END(name=unmatched_stencil)""",
    ],
)
def test_directive_parser_unbalanced_stencil_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])
    directives = collect_directives(fpath)

    with pytest.raises(ParsingException):
        DirectivesParser(directives)
        # todo: introduce more stencil cases
        # todo: exception should be more specific
        # todo: check for error message.

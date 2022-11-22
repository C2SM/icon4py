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

from icon4py.liskov.directives import NoDirectivesFound
from icon4py.liskov.exceptions import DirectiveSyntaxError, ParsingException
from icon4py.liskov.parser import DirectivesParser
from icon4py.testutils.fortran_samples import (
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_STENCIL,
)


def insert_new_lines(fname: Path, lines: list[str]):
    with open(fname, "a") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def test_directive_parser_single_stencil(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    parser = DirectivesParser(fpath)
    directives = parser.parsed_directives

    assert len(directives.stencils) == 1
    assert directives.declare_line == 1
    assert directives.create_line == 2


def test_directive_parser_multiple_stencils(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    parser = DirectivesParser(fpath)
    directives = parser.parsed_directives

    assert len(directives.stencils) == 2
    assert directives.declare_line == 1
    assert directives.create_line == 3


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL FOO_DIRECTIVE",
        "!$DSL BAR_DIRECTIVE",
    ],
)
def test_directive_parser_parsing_exception(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])

    with pytest.raises(ParsingException):
        DirectivesParser(fpath)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL STENCIL START(stencil1, stencil2)",
        "!$DSL DECLARE FOO",
        "!$DSL CREATE DATA",
    ],
)
def test_directive_parser_invalid_directive_syntax(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])

    with pytest.raises(DirectiveSyntaxError):
        DirectivesParser(fpath)


def test_directive_parser_no_directives_found(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=NO_DIRECTIVES_STENCIL)
    parser = DirectivesParser(fpath)
    directives = parser.parsed_directives
    assert isinstance(directives, NoDirectivesFound)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL DECLARE",
        "!$DSL CREATE",
        "!$DSL STENCIL START(mo_nh_diffusion_stencil_06)",
    ],
)
def test_directive_parser_repeated_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])

    with pytest.raises(ParsingException):
        DirectivesParser(fpath)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL DECLARE",
        "!$DSL CREATE",
        "!$DSL STENCIL START(mo_nh_diffusion_stencil_06)",
        "!$DSL STENCIL END(mo_nh_diffusion_stencil_06)",
    ],
)
def test_directive_parser_required_directives(make_f90_tmpfile, directive):
    new = SINGLE_STENCIL.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)

    with pytest.raises(ParsingException):
        DirectivesParser(fpath)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL STENCIL START(unmatched_stencil_10)",
        "!$DSL STENCIL END(unmatched_stencil_23)",
    ],
)
def test_directive_parser_unbalanced_stencil_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])

    with pytest.raises(ParsingException):
        DirectivesParser(fpath)

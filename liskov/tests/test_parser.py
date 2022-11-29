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

from icon4py.liskov.collect import DirectivesCollector
from icon4py.liskov.directives import NoDirectivesFound, RawDirective
from icon4py.liskov.exceptions import DirectiveSyntaxError, ParsingException
from icon4py.liskov.input import (
    BoundsData,
    CreateData,
    DeclareData,
    FieldAssociationData,
    StencilData,
)
from icon4py.liskov.parser import DirectivesParser, ParsedDirectives
from icon4py.testutils.fortran_samples import (
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_STENCIL,
)


def collect_directives(fpath: Path) -> list[RawDirective]:
    collector = DirectivesCollector(fpath)
    return collector.directives


def insert_new_lines(fname: Path, lines: list[str]):
    with open(fname, "a") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def test_directive_parser_single_stencil(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    directives = collect_directives(fpath)
    parser = DirectivesParser(directives)
    parsed = parser.parsed_directives

    # type checks
    assert isinstance(parsed, ParsedDirectives)

    # create checks
    assert isinstance(parsed.create, CreateData)
    assert parsed.create.variables == ["something", "som_field_2"]
    assert parsed.create.startln == 4
    assert parsed.create.endln == 4

    # declare checks
    assert isinstance(parsed.declare, DeclareData)
    assert parsed.declare.declarations == {
        "vt": "(nproma, p_patch%nlev, p_patch%nblks_e)",
        "vn_ie": "(nproma, p_patch%nlevp1, p_patch%nblks_e)",
    }
    assert parsed.declare.startln == 1
    assert parsed.declare.endln == 2

    # stencil checks
    assert isinstance(parsed.stencils[0], StencilData)
    assert isinstance(parsed.stencils[0].bounds, BoundsData)
    assert isinstance(parsed.stencils[0].fields[0], FieldAssociationData)

    assert len(parsed.stencils) == 1
    assert len(parsed.stencils[0].fields) == 4
    assert parsed.stencils[0].bounds.__dict__ == {
        "hlower": "i_startidx",
        "hupper": "i_endidx",
        "vlower": "1",
        "vupper": "nlev",
    }
    assert parsed.stencils[0].startln == 6
    assert parsed.stencils[0].endln == 10
    assert parsed.stencils[0].name == "run_mo_velocity_advection_stencil_07"


def test_directive_parser_multiple_stencils(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    directives = collect_directives(fpath)
    parser = DirectivesParser(directives)
    parsed = parser.parsed_directives

    # type checks
    assert isinstance(parsed, ParsedDirectives)

    # create checks
    assert isinstance(parsed.create, CreateData)
    assert parsed.create.variables == ["something", "som_field_2"]
    assert parsed.create.startln == 4
    assert parsed.create.endln == 4

    # declare checks
    assert isinstance(parsed.declare, DeclareData)
    assert parsed.declare.declarations == {
        "vt": "(nproma, p_patch%nlev, p_patch%nblks_e)",
        "vn_ie": "(nproma, p_patch%nlevp1, p_patch%nblks_e)",
    }
    assert parsed.declare.startln == 1
    assert parsed.declare.endln == 2

    # stencil checks
    assert isinstance(parsed.stencils[1], StencilData)
    assert isinstance(parsed.stencils[1].bounds, BoundsData)
    assert isinstance(parsed.stencils[1].fields[0], FieldAssociationData)

    assert len(parsed.stencils) == 2
    assert len(parsed.stencils[1].fields) == 4
    assert parsed.stencils[1].bounds.__dict__ == {
        "hlower": "i_startidx",
        "hupper": "i_endidx",
        "vlower": "1",
        "vupper": "nlev",
    }
    assert parsed.stencils[1].startln == 24
    assert parsed.stencils[1].endln == 28
    assert parsed.stencils[1].name == "run_mo_velocity_advection_stencil_07"


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


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL START(stencil1, stencil2)",
        "!$DSL DECLARE(somefield; another_field)",
        "!$DSL CREATE(field=field)",
    ],
)
def test_directive_parser_invalid_directive_syntax(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=MULTIPLE_STENCILS)
    insert_new_lines(fpath, [directive])
    directives = collect_directives(fpath)

    with pytest.raises(DirectiveSyntaxError):
        DirectivesParser(directives)


def test_directive_parser_no_directives_found(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=NO_DIRECTIVES_STENCIL)
    directives = collect_directives(fpath)
    parser = DirectivesParser(directives)
    parsed = parser.parsed_directives
    assert isinstance(parsed, NoDirectivesFound)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL CREATE(something; som_field_2)",
        "!$DSL END(name=run_mo_velocity_advection_stencil_07)",
    ],
)
def test_directive_parser_repeated_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    insert_new_lines(fpath, [directive])
    directives = collect_directives(fpath)

    with pytest.raises(ParsingException):
        DirectivesParser(directives)


@pytest.mark.parametrize(
    "directive",
    [
        """!$DSL CREATE(something; som_field_2)""",
        """!$DSL END(name=run_mo_velocity_advection_stencil_07)""",
    ],
)
def test_directive_parser_required_directives(make_f90_tmpfile, directive):
    new = SINGLE_STENCIL.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    directives = collect_directives(fpath)

    with pytest.raises(ParsingException):
        DirectivesParser(directives)


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

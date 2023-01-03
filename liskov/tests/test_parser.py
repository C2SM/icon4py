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
from collections import defaultdict

import pytest
from conftest import insert_new_lines, scan_for_directives
from pytest import mark
from samples.fortran_samples import (
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_STENCIL,
)

from icon4py.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    RepeatedDirectiveError,
    RequiredDirectivesError,
    UnbalancedStencilDirectiveError,
    UnsupportedDirectiveError,
)
from icon4py.liskov.parsing.parse import DirectivesParser
from icon4py.liskov.parsing.types import (
    Create,
    Imports,
    NoDirectivesFound,
    StartStencil,
    TypedDirective,
)


def test_parse_no_input():
    directives = []
    assert DirectivesParser._parse(directives) == defaultdict(list)


@mark.parametrize(
    "directive_type, string, startln, endln, expected_content",
    [
        (Imports(), "IMPORT()", 1, 1, defaultdict(list, {"Import": [{}]})),
        (Create(), "CREATE()", 2, 2, defaultdict(list, {"Create": [{}]})),
        (
            StartStencil(),
            "START(name=mo_nh_diffusion_06; vn=p_patch%p%vn; foo=abc)",
            3,
            4,
            defaultdict(
                list,
                {
                    "Start": [
                        {
                            "name": "mo_nh_diffusion_06",
                            "vn": "p_patch%p%vn",
                            "foo": "abc",
                        }
                    ]
                },
            ),
        ),
    ],
)
def test_parse_single_directive(
    directive_type, string, startln, endln, expected_content
):
    directives = [
        TypedDirective(
            directive_type=directive_type, string=string, startln=startln, endln=endln
        )
    ]
    assert DirectivesParser._parse(directives) == expected_content


@mark.parametrize(
    "stencil, num_directives, num_content",
    [(SINGLE_STENCIL, 5, 5), (MULTIPLE_STENCILS, 9, 5)],
)
def test_file_parsing(make_f90_tmpfile, stencil, num_directives, num_content):
    fpath = make_f90_tmpfile(content=stencil)
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(directives, fpath)
    parsed = parser.parsed_directives

    directives = parsed["directives"]
    content = parsed["content"]

    assert len(directives) == num_directives
    assert len(content) == num_content

    assert isinstance(content, defaultdict)
    assert all([isinstance(d, TypedDirective) for d in directives])


@pytest.mark.parametrize(
    "stencil, directive",
    [
        (SINGLE_STENCIL, "!$DSL FOO_DIRECTIVE()"),
        (MULTIPLE_STENCILS, "!$DSL BAR_DIRECTIVE()"),
    ],
)
def test_unsupported_directives(
    make_f90_tmpfile,
    stencil,
    directive,
):
    fpath = make_f90_tmpfile(content=stencil)
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)

    with pytest.raises(
        UnsupportedDirectiveError,
        match=r"Used unsupported directive\(s\):.",
    ):
        DirectivesParser(directives, fpath)


@pytest.mark.parametrize(
    "stencil, directive",
    [
        (SINGLE_STENCIL, "!$DSL START(stencil1, stencil2)"),
        (MULTIPLE_STENCILS, "!$DSL DECLARE(somefield; another_field)"),
        (SINGLE_STENCIL, "!$DSL IMPORT(field)"),
    ],
)
def test_directive_parser_invalid_directive_syntax(
    make_f90_tmpfile, stencil, directive
):
    fpath = make_f90_tmpfile(content=stencil)
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)

    with pytest.raises(DirectiveSyntaxError, match=r"Error in .+ on line \d+\.\s+."):
        DirectivesParser(directives, fpath)


def test_directive_parser_no_directives_found(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=NO_DIRECTIVES_STENCIL)
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(directives, fpath)
    parsed = parser.parsed_directives
    assert isinstance(parsed, NoDirectivesFound)


@pytest.mark.parametrize(
    "directive",
    [
        "!$DSL IMPORT()",
        "!$DSL END(name=mo_nh_diffusion_stencil_06)",
        "!$DSL CREATE()",
    ],
)
def test_directive_parser_repeated_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)

    with pytest.raises(
        RepeatedDirectiveError,
        match="Found same directive more than once in the following directives:\n",
    ):
        DirectivesParser(directives, fpath)


@pytest.mark.parametrize(
    "directive",
    [
        """!$DSL IMPORT()""",
        """!$DSL CREATE()""",
        """!$DSL END(name=mo_nh_diffusion_stencil_06)""",
    ],
)
def test_directive_parser_required_directives(make_f90_tmpfile, directive):
    new = SINGLE_STENCIL.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    directives = scan_for_directives(fpath)

    with pytest.raises(
        RequiredDirectivesError,
        match=r"Missing required directive of type (\w*) in source.",
    ):
        DirectivesParser(directives, fpath)


@pytest.mark.parametrize(
    "stencil, directive",
    [
        (MULTIPLE_STENCILS, "!$DSL END(name=mo_nh_diffusion_stencil_06)"),
        (MULTIPLE_STENCILS, "!$DSL END(name=mo_solve_nonhydro_stencil_16)"),
    ],
)
def test_directive_parser_unbalanced_stencil_directives(
    make_f90_tmpfile, stencil, directive
):
    new = stencil.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    directives = scan_for_directives(fpath)

    with pytest.raises(
        UnbalancedStencilDirectiveError,
        match=r"Found (\d*) unbalanced START or END directive(s).\n",
    ):
        DirectivesParser(directives, fpath)

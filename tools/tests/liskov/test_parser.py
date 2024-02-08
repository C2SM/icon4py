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
from pytest import mark

import icon4pytools.liskov.parsing.parse as ts
from icon4pytools.liskov.parsing.exceptions import UnsupportedDirectiveError
from icon4pytools.liskov.parsing.parse import DirectivesParser

from .conftest import insert_new_lines, scan_for_directives
from .fortran_samples import (
    MULTIPLE_STENCILS,
    NO_DIRECTIVES_STENCIL,
    SINGLE_FUSED,
    SINGLE_STENCIL,
    SINGLE_STENCIL_WITH_COMMENTS,
)


def test_parse_no_input():
    directives = []
    assert DirectivesParser._parse(directives) == defaultdict(list)


@mark.parametrize(
    "directive, string, startln, endln, expected_content",
    [
        (
            ts.Imports("IMPORTS()", 1, 1),
            "IMPORTS()",
            1,
            1,
            defaultdict(list, {"Imports": [{}]}),
        ),
        (
            ts.StartCreate("START CREATE(extra_fields=foo)", 2, 2),
            "START CREATE()",
            2,
            2,
            defaultdict(list, {"StartCreate": [{"extra_fields": "foo"}]}),
        ),
        (
            ts.StartStencil(
                "START STENCIL(name=mo_nh_diffusion_06; vn=p_patch%p%vn; foo=abc)", 3, 4
            ),
            "START STENCIL(name=mo_nh_diffusion_06; vn=p_patch%p%vn; foo=abc)",
            3,
            4,
            defaultdict(
                list,
                {
                    "StartStencil": [
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
def test_parse_single_directive(directive, string, startln, endln, expected_content):
    directives = [directive]
    assert DirectivesParser._parse(directives) == expected_content


@mark.parametrize(
    "stencil, num_directives, num_content",
    [
        (SINGLE_STENCIL, 10, 9),
        (SINGLE_STENCIL_WITH_COMMENTS, 9, 8),
        (MULTIPLE_STENCILS, 11, 7),
        (SINGLE_FUSED, 9, 7),
    ],
)
def test_file_parsing(make_f90_tmpfile, stencil, num_directives, num_content):
    fpath = make_f90_tmpfile(content=stencil)
    opath = fpath.with_suffix(".gen")
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)
    parsed = parser(directives)

    directives = parsed["directives"]
    content = parsed["content"]

    assert len(directives) == num_directives
    assert len(content) == num_content

    assert isinstance(content, defaultdict)
    assert all([isinstance(d, ts.ParsedDirective) for d in directives])


def test_directive_parser_no_directives_found(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=NO_DIRECTIVES_STENCIL)
    opath = fpath.with_suffix(".gen")
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)
    with pytest.raises(SystemExit):
        parser(directives)


@mark.parametrize(
    "stencil, directive",
    [
        (SINGLE_STENCIL, "!$DSL FOO()"),
        (SINGLE_STENCIL_WITH_COMMENTS, "!$DSL FOO()"),
        (MULTIPLE_STENCILS, "!$DSL BAR()"),
    ],
)
def test_unsupported_directives(
    make_f90_tmpfile,
    stencil,
    directive,
):
    fpath = make_f90_tmpfile(content=stencil)
    opath = fpath.with_suffix(".gen")
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)

    with pytest.raises(
        UnsupportedDirectiveError,
        match=r"Used unsupported directive\(s\):.",
    ):
        parser(directives)

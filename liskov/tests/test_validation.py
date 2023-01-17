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
from conftest import insert_new_lines, scan_for_directives
from pytest import mark
from samples.fortran_samples import MULTIPLE_STENCILS, SINGLE_STENCIL

from icon4py.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    RepeatedDirectiveError,
    RequiredDirectivesError,
    UnbalancedStencilDirectiveError,
)
from icon4py.liskov.parsing.parse import DirectivesParser


@mark.parametrize(
    "stencil, directive",
    [
        (
            MULTIPLE_STENCILS,
            "!$DSL START STENCIL(name=foo)\n!$DSL END STENCIL(name=moo)",
        ),
        (MULTIPLE_STENCILS, "!$DSL END STENCIL(name=foo)"),
    ],
)
def test_directive_parser_unbalanced_stencil_directives(
    make_f90_tmpfile, stencil, directive
):
    fpath = make_f90_tmpfile(stencil + directive)
    directives = scan_for_directives(fpath)

    with pytest.raises(UnbalancedStencilDirectiveError):
        DirectivesParser(directives, fpath)


# todo: improve validation tests
@mark.parametrize(
    "stencil, directive",
    [
        (SINGLE_STENCIL, "!$DSL START STENCIL(stencil1, stencil2)"),
        (MULTIPLE_STENCILS, "!$DSL DECLARE(somefield; another_field)"),
        (SINGLE_STENCIL, "!$DSL IMPORTS(field)"),
        (SINGLE_STENCIL, "!$DSL IMPORTS())"),
        # (SINGLE_STENCIL, "!$DSL START CREATE(;)"),  # todo: single not allowed
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


@mark.parametrize(
    "directive",
    [
        "!$DSL IMPORTS()",
        "!$DSL END STENCIL(name=mo_nh_diffusion_stencil_06)",
        "!$DSL START CREATE()",
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


@mark.parametrize(
    "directive",
    [
        """!$DSL IMPORTS()""",
        """!$DSL START CREATE()""",
        """!$DSL END STENCIL(name=mo_nh_diffusion_stencil_06)""",
    ],
)
def test_directive_parser_required_directives(make_f90_tmpfile, directive):
    new = SINGLE_STENCIL.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    directives = scan_for_directives(fpath)

    with pytest.raises(
        RequiredDirectivesError,
        match=r"Missing required directive of type (\w.*) in source.",
    ):
        DirectivesParser(directives, fpath)

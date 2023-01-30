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
from icon4py.liskov.parsing.types import (
    Declare,
    Imports,
    StartCreate,
    StartStencil,
)
from icon4py.liskov.parsing.validation import DirectiveSyntaxValidator


@mark.parametrize(
    "stencil, directive",
    [
        (
            MULTIPLE_STENCILS,
            "!$DSL START STENCIL(name=foo)\n!$DSL END STENCIL(name=bar)",
        ),
        (MULTIPLE_STENCILS, "!$DSL END STENCIL(name=foo)"),
    ],
)
def test_directive_semantics_validation_unbalanced_stencil_directives(
    make_f90_tmpfile, stencil, directive
):
    fpath = make_f90_tmpfile(stencil + directive)
    directives = scan_for_directives(fpath)

    with pytest.raises(UnbalancedStencilDirectiveError):
        DirectivesParser(directives, fpath)


@mark.parametrize(
    "directive",
    (
        [StartStencil("!$DSL START STENCIL(name=foo, x=bar)", 0, 0)],
        [StartStencil("!$DSL START STENCIL(name=foo, x=bar;)", 0, 0)],
        [StartStencil("!$DSL START STENCIL(name=foo, x=bar;", 0, 0)],
        [Declare("!$DSL DECLARE(name=foo; bar)", 0, 0)],
        [Imports("!$DSL IMPORTS(foo)", 0, 0)],
        [Imports("!$DSL IMPORTS())", 0, 0)],
        [StartCreate("!$DSL START CREATE(;)", 0, 0)],
    ),
)
def test_directive_syntax_validator(directive):
    validator = DirectiveSyntaxValidator("test")
    with pytest.raises(DirectiveSyntaxError, match=r"Error in .+ on line \d+\.\s+."):
        validator.validate(directive)


@mark.parametrize(
    "directive",
    [
        "!$DSL IMPORTS()",
        "!$DSL START CREATE()",
    ],
)
def test_directive_semantics_validation_repeated_directives(
    make_f90_tmpfile, directive
):
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
        "!$DSL START STENCIL(name=mo_nh_diffusion_stencil_06)\n!$DSL END STENCIL(name=mo_nh_diffusion_stencil_06)"
    ],
)
def test_directive_semantics_validation_repeated_stencil(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL)
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)
    DirectivesParser(directives, fpath)


@mark.parametrize(
    "directive",
    [
        """!$DSL IMPORTS()""",
        """!$DSL START CREATE()""",
        """!$DSL END STENCIL(name=mo_nh_diffusion_stencil_06)""",
    ],
)
def test_directive_semantics_validation_required_directives(
    make_f90_tmpfile, directive
):
    new = SINGLE_STENCIL.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    directives = scan_for_directives(fpath)

    with pytest.raises(
        RequiredDirectivesError,
        match=r"Missing required directive of type (\w.*) in source.",
    ):
        DirectivesParser(directives, fpath)

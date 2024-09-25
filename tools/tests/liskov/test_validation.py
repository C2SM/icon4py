# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from pytest import mark

from icon4pytools.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    RepeatedDirectiveError,
    RequiredDirectivesError,
    UnbalancedStencilDirectiveError,
)
from icon4pytools.liskov.parsing.parse import Declare, DirectivesParser, Imports, StartStencil
from icon4pytools.liskov.parsing.validation import DirectiveSyntaxValidator

from .conftest import insert_new_lines, scan_for_directives
from .fortran_samples import (
    MULTIPLE_FUSED,
    MULTIPLE_STENCILS,
    SINGLE_FUSED,
    SINGLE_STENCIL_WITH_COMMENTS,
)


@mark.parametrize(
    "stencil, directive",
    [
        (
            MULTIPLE_STENCILS,
            "!$DSL START STENCIL(name=foo)\n!$DSL END STENCIL(name=bar)",
        ),
        (MULTIPLE_STENCILS, "!$DSL END STENCIL(name=foo)"),
        (
            MULTIPLE_FUSED,
            "!$DSL START STENCIL(name=foo)\n!$DSL END STENCIL(name=bar)",
        ),
        (MULTIPLE_FUSED, "!$DSL END STENCIL(name=foo)"),
        (
            MULTIPLE_FUSED,
            "!$DSL START FUSED STENCIL(name=foo)\n!$DSL END FUSED STENCIL(name=bar)",
        ),
        (MULTIPLE_FUSED, "!$DSL END FUSED STENCIL(name=foo)"),
    ],
)
def test_directive_semantics_validation_unbalanced_stencil_directives(
    make_f90_tmpfile, stencil, directive
):
    fpath = make_f90_tmpfile(stencil + directive)
    opath = fpath.with_suffix(".gen")
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)

    with pytest.raises(UnbalancedStencilDirectiveError):
        parser(directives)


@mark.parametrize(
    "stencil, directive",
    [
        (
            SINGLE_STENCIL_WITH_COMMENTS,
            "!$DSL START STENCIL( name = foo ; x = bar )\n!$DSL END STENCIL(name = foo)",
        ),
        (
            SINGLE_STENCIL_WITH_COMMENTS,
            "!$DSL START FUSED STENCIL( name = foo ; x = bar )\n!$DSL END FUSED STENCIL(name = foo)",
        ),
    ],
)
def test_directive_semantics_validation_allow_whitespaces_in_name_arg(
    make_f90_tmpfile, stencil, directive
):
    fpath = make_f90_tmpfile(stencil + directive)
    opath = fpath.with_suffix(".gen")
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)
    parser(directives)


@mark.parametrize(
    "directive",
    (
        [StartStencil("!$DSL START STENCIL(name=foo, x=bar)", 0, 0)],
        [StartStencil("!$DSL START STENCIL(name=foo, x=bar;)", 0, 0)],
        [StartStencil("!$DSL START STENCIL(name=foo, x=bar;", 0, 0)],
        [Declare("!$DSL DECLARE(name=foo; bar)", 0, 0)],
        [Imports("!$DSL IMPORTS(foo)", 0, 0)],
        [Imports("!$DSL IMPORTS())", 0, 0)],
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
    ],
)
def test_directive_semantics_validation_repeated_directives(make_f90_tmpfile, directive):
    fpath = make_f90_tmpfile(content=SINGLE_STENCIL_WITH_COMMENTS)
    opath = fpath.with_suffix(".gen")
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)

    with pytest.raises(
        RepeatedDirectiveError,
        match="Found same directive more than once in the following directives:\n",
    ):
        parser(directives)


@mark.parametrize(
    "stencil, directive",
    [
        (
            SINGLE_STENCIL_WITH_COMMENTS,
            "!$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary)\n!$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary)",
        ),
        (
            SINGLE_FUSED,
            "!$DSL START FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence)\n!$DSL END FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence)",
        ),
    ],
)
def test_directive_semantics_validation_repeated_stencil(make_f90_tmpfile, stencil, directive):
    fpath = make_f90_tmpfile(content=stencil)
    opath = fpath.with_suffix(".gen")
    insert_new_lines(fpath, [directive])
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)
    parser(directives)


@mark.parametrize(
    "directive",
    [
        """!$DSL IMPORTS()""",
        """!$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; noprofile=True)""",
    ],
)
def test_directive_semantics_validation_required_directives(make_f90_tmpfile, directive):
    new = SINGLE_STENCIL_WITH_COMMENTS.replace(directive, "")
    fpath = make_f90_tmpfile(content=new)
    opath = fpath.with_suffix(".gen")
    directives = scan_for_directives(fpath)
    parser = DirectivesParser(fpath, opath)

    with pytest.raises(
        RequiredDirectivesError,
        match=r"Missing required directive of type (\w.*) in source.",
    ):
        parser(directives)

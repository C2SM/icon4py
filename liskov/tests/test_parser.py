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

from icon4py.liskov.parser import DirectivesParser
from icon4py.testutils.fortran_samples import MULTIPLE_STENCILS, SINGLE_STENCIL


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


# todo: add tests for invalid preprocessor directives (should raise DirectiveSyntaxError)

# todo: add test for case where no directives are found (should return NoDirectives)

# todo: add test for use of an unsupported directive (should raise ParsingException)

# todo: add test for multiple unsupported directives (should raise ParsingException)

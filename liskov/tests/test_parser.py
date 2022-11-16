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
from icon4py.testutils.fortran_samples import SIMPLE_STENCIL


def test_directive_parser(make_f90_tmpfile):
    fpath = make_f90_tmpfile(content=SIMPLE_STENCIL)
    parser = DirectivesParser(fpath)
    parser()

    assert len(parser.directives) == 2


# todo: test simple directive parsing (start and end directives), F90 template is written to file, then parsed.

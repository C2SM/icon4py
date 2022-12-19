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
import tempfile
from pathlib import Path

from pytest import mark
from samples.fortran_samples import DIRECTIVES_SAMPLE, NO_DIRECTIVES_STENCIL

from icon4py.liskov.parsing.scan import DirectivesScanner
from icon4py.liskov.parsing.types import RawDirective


def scan_tempfile(string: str, expected: list):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(string.encode())
        tmp.flush()
        scanner = DirectivesScanner(Path(tmp.name))
        assert scanner.directives == expected


@mark.parametrize(
    "string,expected",
    [
        (NO_DIRECTIVES_STENCIL, []),
        (
            DIRECTIVES_SAMPLE,
            [
                RawDirective("!$DSL IMPORT()\n", 0, 0),
                RawDirective("!$DSL CREATE()\n", 2, 2),
                RawDirective("!$DSL DECLARE(vn=p_patch%vn; vn2=p_patch%vn2)\n", 4, 4),
                RawDirective(
                    "!$DSL START(name=mo_nh_diffusion_06; vn=p_patch%vn; &\n!$DSL       a=a; b=c)\n",
                    6,
                    7,
                ),
                RawDirective("!$DSL END(name=mo_nh_diffusion_06)\n", 9, 9),
                RawDirective(
                    "!$DSL START(name=mo_nh_diffusion_07; xn=p_patch%xn)\n", 11, 11
                ),
                RawDirective("!$DSL END(name=mo_nh_diffusion_07)\n", 13, 13),
                RawDirective("!$DSL UNKNOWN_DIRECTIVE()\n", 15, 15),
            ],
        ),
    ],
)
def test_directives_scanning(string, expected):
    scan_tempfile(string, expected)

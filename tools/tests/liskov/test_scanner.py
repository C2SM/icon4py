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
import string
import tempfile
from pathlib import Path

import pytest
from icon4pytools.liskov.parsing.exceptions import DirectiveSyntaxError
from icon4pytools.liskov.parsing.scan import DirectivesScanner
from icon4pytools.liskov.parsing.types import RawDirective
from pytest import mark

from icon4py.testutils.liskov.fortran_samples import (
    DIRECTIVES_SAMPLE,
    NO_DIRECTIVES_STENCIL,
)


ALLOWED_EOL_CHARS = [")", "&"]


def scan_tempfile(string: str):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(string.encode())
        tmp.flush()
        scanner = DirectivesScanner(Path(tmp.name))
        return scanner()


def special_char():
    def special_chars_generator():
        for char in string.punctuation:
            yield char

    return special_chars_generator()


@mark.parametrize(
    "string,expected",
    [
        (NO_DIRECTIVES_STENCIL, []),
        (
            DIRECTIVES_SAMPLE,
            [
                RawDirective("!$DSL IMPORTS()\n", 0, 0),
                RawDirective("!$DSL START CREATE()\n", 2, 2),
                RawDirective("!$DSL DECLARE(vn=p_patch%vn; vn2=p_patch%vn2)\n", 4, 4),
                RawDirective(
                    "!$DSL START STENCIL(name=mo_nh_diffusion_06; vn=p_patch%vn; &\n!$DSL       a=a; b=c)\n",
                    6,
                    7,
                ),
                RawDirective("!$DSL END STENCIL(name=mo_nh_diffusion_06)\n", 9, 9),
                RawDirective(
                    "!$DSL START STENCIL(name=mo_nh_diffusion_07; xn=p_patch%xn)\n",
                    11,
                    11,
                ),
                RawDirective("!$DSL END STENCIL(name=mo_nh_diffusion_07)\n", 13, 13),
                RawDirective("!$DSL UNKNOWN_DIRECTIVE()\n", 15, 15),
                RawDirective("!$DSL END CREATE()\n", 16, 16),
            ],
        ),
    ],
)
def test_directives_scanning(string, expected):
    scanned = scan_tempfile(string)
    assert scanned == expected


@pytest.mark.parametrize("special_char", special_char())
def test_directive_eol(special_char):
    if special_char in ALLOWED_EOL_CHARS:
        pytest.skip()
    else:
        directive = "!$DSL IMPORT(" + special_char
        with pytest.raises(DirectiveSyntaxError):
            scan_tempfile(directive)


def test_directive_unclosed():
    directive = "!$DSL IMPORT(&\n!CALL foo()"
    with pytest.raises(DirectiveSyntaxError):
        scan_tempfile(directive)

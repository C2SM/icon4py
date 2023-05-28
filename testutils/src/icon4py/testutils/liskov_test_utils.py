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

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.parsing.scan import DirectivesScanner


def scan_for_directives(fpath: Path) -> list[ts.RawDirective]:
    collector = DirectivesScanner(fpath)
    return collector()


def insert_new_lines(fname: Path, lines: list[str]) -> None:
    """Append new lines into file."""
    with open(fname, "a") as f:
        for ln in lines:
            f.write(f"{ln}\n")

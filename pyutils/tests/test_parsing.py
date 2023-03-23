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

from icon4py.serialisation.interface import SerialisationInterface
from icon4py.serialisation.parse import GranuleParser


# todo: test for correct parsing
def test_granule_parsing():
    root_dir = Path(__file__).parent
    path = Path(f"{root_dir}/samples/granule_example.f90")
    parser = GranuleParser(path)
    parsed = parser.parse()
    assert parsed == SerialisationInterface


# todo: add test for intrinsic types

# todo: add test for derived types

# todo: add test for post declaration line number

# todo: add test for end of subroutine line number

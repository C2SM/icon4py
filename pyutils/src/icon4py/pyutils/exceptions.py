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

from typing import List


class MultipleFieldOperatorException(Exception):
    def __init___(self, stencil_name: str) -> None:
        Exception.__init__(
            self,
            f"{stencil_name} is currently not supported as it contains multiple field operators.",
        )


class InvalidConnectivityException(Exception):
    def __init___(self, location_chain: List[str]) -> None:
        Exception.__init__(
            self,
            f"Connectivity identifier must be one of [C, E, V, O], provided: {location_chain}",
        )

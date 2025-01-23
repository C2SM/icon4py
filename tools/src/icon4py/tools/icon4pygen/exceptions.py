# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List


class InvalidConnectivityException(Exception):
    def __init___(self, location_chain: List[str]) -> None:
        Exception.__init__(
            self,
            f"Connectivity identifier must be one of [C, E, V, O], provided: {location_chain}",
        )

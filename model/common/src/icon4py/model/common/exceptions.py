# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


class InvalidConfigError(Exception):
    pass


class IncompleteStateError(Exception):
    def __init__(self, field_name):
        super().__init__(f"Field '{field_name}' is missing in state.")

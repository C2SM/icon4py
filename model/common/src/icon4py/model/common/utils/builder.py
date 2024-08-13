# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


def builder(func):
    """Use as decorator on builder functions."""

    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return wrapper

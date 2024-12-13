# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.testing.helpers import backend, grid
from icon4py.model.testing.pytest_config import *  # noqa: F401

__all__ = [
    # imported fixtures:
    "backend",
    "grid"
]

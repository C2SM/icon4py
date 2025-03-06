# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.testing.helpers import connectivities_as_numpy


# ruff: noqa: F405
# Make sure custom icon4py pytest hooks are loaded
try:
    import sys

    _ = sys.modules["icon4py.model.testing.pytest_config"]
except KeyError:
    from icon4py.model.testing.pytest_config import *  # noqa: F403 [undefined-local-with-import-star]


__all__ = [
    # imported fixtures:
    "connectivities_as_numpy",
    "grid",
    "backend",
]

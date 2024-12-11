# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util

import pytest


if not importlib.util.find_spec("xarray"):
    pytest.fail(
        "Optional icon4py-common[io] dependencies are missing. Please install them using `pip install icon4py-common[io]`."
    )
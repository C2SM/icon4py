# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os


_ACTIVE = os.environ.get("ICON4PY_COMPROF") == "1"
if _ACTIVE:
    _plugs = os.environ.get("PYTEST_PLUGINS", "")
    _plugs = ",".join(p for p in _plugs.split(",") if p)
    os.environ["PYTEST_PLUGINS"] = f"{_plugs},comprof_plugin" if _plugs else "comprof_plugin"

from compcore import install  # noqa: E402


install()

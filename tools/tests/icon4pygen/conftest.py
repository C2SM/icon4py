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
from icon4pytools.common import ICON4PY_MODEL_QUALIFIED_NAME


def get_stencil_module_path(stencil_module: str, stencil_name: str) -> str:
    return f"{ICON4PY_MODEL_QUALIFIED_NAME}.{stencil_module}.{stencil_name}:{stencil_name}"

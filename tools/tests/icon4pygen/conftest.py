# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4pytools.common import ICON4PY_MODEL_QUALIFIED_NAME


def get_stencil_module_path(stencil_module: str, stencil_name: str) -> str:
    return f"{ICON4PY_MODEL_QUALIFIED_NAME}.{stencil_module}.{stencil_name}:{stencil_name}"

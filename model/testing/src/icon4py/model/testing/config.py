# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib


def _env_flag_to_bool(name: str, default: bool) -> bool:
    """Convert environment variable string variable to a bool value."""
    flag_value = os.environ.get(name, None)
    if flag_value is None:
        return default
    match flag_value.lower():
        case "0" | "false" | "off":
            return False
        case "1" | "true" | "on":
            return True
        case _:
            raise ValueError(
                "Invalid ICON4Py environment flag value: use '0 | false | off' or '1 | true | on'."
            )

def _env_path(name: str, default: pathlib.Path) -> pathlib.Path:
    value = os.environ.get(name)
    return pathlib.Path(value) if value is not None else default


ENABLE_GRID_DOWNLOAD: bool = _env_flag_to_bool("ICON4PY_ENABLE_GRID_DOWNLOAD", True)
ENABLE_TESTDATA_DOWNLOAD: bool = _env_flag_to_bool("ICON4PY_ENABLE_TESTDATA_DOWNLOAD", True)

_TEST_DATA_DIR: str = "testdata"
TEST_DATA_PATH: pathlib.Path = _env_path(
    "ICON4PY_TEST_DATA_PATH",
    pathlib.Path(__file__).parents[6] / _TEST_DATA_DIR,
)

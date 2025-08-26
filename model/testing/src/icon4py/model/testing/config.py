# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os


def env_flag_to_bool(name: str, default: bool) -> bool:
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


ENABLE_GRID_DOWNLOAD: bool = env_flag_to_bool("ICON4PY_ENABLE_GRID_DOWNLOAD", True)
ENABLE_TESTDATA_DOWNLOAD: bool = env_flag_to_bool("ICON4PY_ENABLE_TESTDATA_DOWNLOAD", True)
TEST_DATA_PATH: str | None = os.environ.get("ICON4PY_TEST_DATA_PATH", None)

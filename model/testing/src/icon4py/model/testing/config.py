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
import tempfile

from icon4py.model.common.utils import env


def _project_root() -> pathlib.Path:
    for path in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]:
        if (path / ".git").exists():
            return path
    # fallback to hardcoded relative path
    return pathlib.Path(__file__).parents[6]


def _default_download_cache() -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / "icon4py_download_cache"


ENABLE_GRID_DOWNLOAD: bool = env.flag_to_bool("ICON4PY_ENABLE_GRID_DOWNLOAD", True)
ENABLE_TESTDATA_DOWNLOAD: bool = env.flag_to_bool("ICON4PY_ENABLE_TESTDATA_DOWNLOAD", True)
TEST_DATA_PATH: pathlib.Path = env.path("ICON4PY_TEST_DATA_PATH", _project_root() / "testdata")
DALLCLOSE_PRINT_INSTEAD_OF_FAIL: bool = env.flag_to_bool(
    "ICON4PY_DALLCLOSE_PRINT_INSTEAD_OF_FAIL", False
)
# When set to a file path, 'assert_dallclose' records the measured max absolute/relative
# differences (instead of asserting) so that tolerances can be measured and tightened automatically.
RECORD_TOLERANCES_PATH: pathlib.Path | None = (
    env.path("ICON4PY_RECORD_TOLERANCES", pathlib.Path())
    if "ICON4PY_RECORD_TOLERANCES" in os.environ
    else None
)
# When enabled, 'assert_dallclose' emits a non-failing warning whenever a passed tolerance is much
# larger than the measured difference, flagging tolerances that have become too loose.
TOLERANCE_DRIFT_WARN: bool = env.flag_to_bool("ICON4PY_TOLERANCE_DRIFT_WARN", False)
DOWNLOAD_CACHE_PATH: pathlib.Path = env.path("ICON4PY_DOWNLOAD_CACHE", _default_download_cache())
DRIVER_LOGGING_LEVEL: str = os.environ.get("ICON4PY_DRIVER_LOGGING_LEVEL", "debug")

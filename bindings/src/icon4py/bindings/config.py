# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Final, Literal

from icon4py.model.common.utils import env


WAIT_FOR_COMPILATION: bool = env.flag_to_bool("ICON4PY_WAIT_FOR_COMPILATION", False)
"""Wait in granule initialization until jit compilation is complete."""


def _icon_precision() -> Literal["double", "mixed"]:
    value = os.environ.get("ICON4PY_BINDINGS_ICON_PRECISION", "double").lower()
    if value not in ("double", "mixed"):
        raise ValueError(
            f"Invalid value {value!r} for environment variable 'ICON4PY_BINDINGS_ICON_PRECISION': "
            "use 'double' or 'mixed'."
        )
    return value  # type: ignore[return-value] # narrowed by the check above


ICON_PRECISION: Final = _icon_precision()
"""Precision ICON itself was built with: 'double' (`wp=vp=dp`) or 'mixed' (`wp=dp`, `vp=sp`).

This is the precision of the Fortran caller, not of icon4py (see `ICON4PY_FLOAT_PRECISION`).
When the bindings are generated it fixes the C kind of every `REAL(vp)` argument, so it must
match ICON's `--enable-mixed-precision`; a mismatch there fails when ICON compiles. When they
run, every array ICON passes is checked against the kinds this setting implies, so a library
generated with a different setting raises instead of computing.
"""

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TypeVar


_T = TypeVar("_T")


def list_to_value(obj: list[_T] | _T) -> _T:
    # Some parameters are allocated as `max_dom`-sized lists, with one value
    # per domain. ICON4Py (for now) only runs on one domain.
    # Most parameters have the same value for all elements, others (such as
    # num_levels) have a default value different from domain[0].
    # Tracers are an even different case where there is one value per tracer,
    # but with the current version of ICON4Py all tracers get the same config.
    return obj[0] if isinstance(obj, list) else obj


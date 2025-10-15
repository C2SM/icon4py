# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.testing import definitions


if TYPE_CHECKING:
    pass


@pytest.fixture(
    params=[definitions.Grids.MCH_OPR_R04B07_DOMAIN01],
    ids=lambda r: r.name,
)
# TODO (Yilu) change to the right grids for benchmarks
def benchmark_grid(request: pytest.FixtureRequest) -> definitions.GridDescription:
    """Default parametrization for benchmark testing.

    The default parametrization is often overwritten for specific tests."""
    return request.param

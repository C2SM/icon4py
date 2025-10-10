# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pkgutil
from typing import TYPE_CHECKING

import gt4py.next.typing as gtx_typing
import pytest

import icon4py.model.common.decomposition.definitions as decomposition
from icon4py.model.common import model_backends
from icon4py.model.common.constants import RayleighType
from icon4py.model.common.grid import base as base_grid
from icon4py.model.testing import (
    config,
    data_handling as data,
    datatest_utils as dt_utils,
    definitions,
    locking,
)


if TYPE_CHECKING:
    import pathlib

    from icon4py.model.testing import serialbox


@pytest.fixture(
    params=[definitions.Grids.MCH_OPR_R04B07_DOMAIN01],
    ids=lambda r: r.name,
)
def benchmark_grid(request: pytest.FixtureRequest) -> definitions.GridDescription:
    """Default parametrization for benchmark testing.

    The default parametrization is often overwritten for specific tests."""
    return request.param

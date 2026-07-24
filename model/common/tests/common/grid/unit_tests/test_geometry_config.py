# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from icon4py.model.common.config import reader
from icon4py.model.common.grid import geometry_config


def test_read_defaults() -> None:
    assert reader.read("{}", geometry_config.GeometryConfig).use_analytical_means


def test_read_explicit() -> None:
    assert not reader.read(
        "use_analytical_means: false", geometry_config.GeometryConfig
    ).use_analytical_means

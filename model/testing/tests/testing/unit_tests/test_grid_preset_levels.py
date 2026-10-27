# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import SimpleNamespace

import pytest

from icon4py.model.testing.fixtures import stencil_tests


@pytest.mark.parametrize(
    "name, default",
    [
        ("icon_regional", 40),
        ("icon_global", 40),
        ("icon_benchmark_regional", 80),
        ("icon_benchmark_global", 80),
    ],
)
@pytest.mark.parametrize("suffix, explicit", [("", None), (":", None), (":40", 40), (":120", 120)])
def test_grid_option_reaches_preset(monkeypatch, name, default, suffix, explicit):
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda key: name + suffix))
    preset, levels = stencil_tests._evaluate_grid_option(request)
    assert levels == explicit
    captured = {}

    def load_grid(identifier, **kwargs):
        captured.update(kwargs)
        return "grid manager"

    monkeypatch.setattr(stencil_tests.grid_utils, "get_grid_manager_from_identifier", load_grid)
    assert (
        stencil_tests._get_grid_manager_from_preset(preset, num_levels=levels, allocator=None)
        == "grid manager"
    )
    assert captured["num_levels"] == (default if explicit is None else explicit)

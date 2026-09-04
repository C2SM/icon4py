# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from icon4py.model.testing import definitions as test_defs, grid_utils


class TestResolveGridDescription:
    def test_resolves_preset(self) -> None:
        assert grid_utils.resolve_grid_description("icon_global") is test_defs.Grids.R02B04_GLOBAL

    def test_resolves_grids_attribute_name(self) -> None:
        assert grid_utils.resolve_grid_description("R02B06_GLOBAL") is test_defs.Grids.R02B06_GLOBAL

    def test_resolves_grid_description_by_name(self) -> None:
        assert (
            grid_utils.resolve_grid_description(test_defs.Grids.R02B06_GLOBAL.name)
            is test_defs.Grids.R02B06_GLOBAL
        )

    def test_strips_level_suffix(self) -> None:
        assert (
            grid_utils.resolve_grid_description("icon_global:80") is test_defs.Grids.R02B04_GLOBAL
        )

    def test_unknown_name_raises_usage_error(self) -> None:
        with pytest.raises(pytest.UsageError, match="Unknown grid 'no_such_grid'"):
            grid_utils.resolve_grid_description("no_such_grid")

    def test_custom_presets_override_default(self) -> None:
        custom = {"custom": test_defs.Grids.R02B04_GLOBAL}
        assert (
            grid_utils.resolve_grid_description("custom", presets=custom)
            is test_defs.Grids.R02B04_GLOBAL
        )

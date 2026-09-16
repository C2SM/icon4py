# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
from collections.abc import Callable

import pytest
import textual
import textual.widgets

from icon4py.model.common.config import config_doc
from icon4py.model.driver.config import ExperimentConfig


@pytest.mark.level("integration")
def test_initial(snap_compare: Callable):
    """Test the initial view."""
    app = config_doc.ConfigDocApp(ExperimentConfig)
    assert snap_compare(app, terminal_size=(160, 40))


@pytest.mark.level("integration")
@pytest.mark.asyncio
async def test_select_everything():
    """Test that there are no crashes when selecting all of the config options."""
    app = config_doc.ConfigDocApp(ExperimentConfig)
    async with app.run_test() as pilot:
        tree = app.query_one("#tree", expect_type=textual.widgets.Tree)
        assert tree.cursor_line == 0
        i = 1
        while tree.validate_cursor_line(i) == i:
            await pilot.press("down", "enter")
            assert tree.cursor_line == i
            i += 1

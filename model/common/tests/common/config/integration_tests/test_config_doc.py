# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum

import pytest
import textual
import textual.widgets

from icon4py.model.common.config import config_doc


@pytest.mark.asyncio
async def test_initial():
    """Test the initial view."""
    app = config_doc.ConfigDocApp()
    async with app.run_test() as _:
        tree = app.query_one("#tree", expect_type=textual.widgets.Tree)
        assert str(tree.root.label) == "icon4py-config.yml"
        assert len(tree.root.children) > 10
        table = app.query_one("#info-table", expect_type=textual.widgets.DataTable)
        assert table.get_cell_at((0, 1)) == "This is the top-level of the config file."


@pytest.mark.asyncio
async def test_select_everything():
    """Test that there are no crashes when selecting all of the config options."""
    app = config_doc.ConfigDocApp()
    async with app.run_test() as pilot:
        tree = app.query_one("#tree", expect_type=textual.widgets.Tree)
        assert tree.cursor_line == 0
        i = 1
        while tree.validate_cursor_line(i) == i:
            await pilot.press("down", "enter")
            assert tree.cursor_line == i
            i += 1

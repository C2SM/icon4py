# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from unittest import mock

import pytest

from icon4py.model.testing.pytest_hooks import pytest_collection_modifyitems


def _make_item(name: str, level: str | None) -> mock.Mock:
    """Build a fake pytest item with the given ``level`` marker, or none."""
    item = mock.Mock()
    item.name = name
    if level is None:
        item.get_closest_marker.return_value = None
    else:
        item.get_closest_marker.return_value = pytest.mark.level(level).mark
    return item


def _make_config(level: str) -> mock.MagicMock:
    """Build a fake pytest config whose ``--level`` option returns ``level``."""
    config = mock.MagicMock()
    config._mpi_scheduler = None
    config.getoption.return_value = level
    return config


def test_level_deselection_preserves_collection_order() -> None:
    """``--level`` deselection must keep matched items in original order.

    Regression test for the previous set-based implementation whose iteration
    order was non-deterministic. Items without a ``level`` marker are treated
    as unit level.
    """
    items = [
        _make_item("test_a", level="unit"),
        _make_item("test_b", level="integration"),
        _make_item("test_c", level=None),
        _make_item("test_d", level="unit"),
        _make_item("test_e", level="validation"),
    ]
    removed_expected = [items[1], items[4]]
    config = _make_config(level="unit")

    pytest_collection_modifyitems(config, items)

    assert [item.name for item in items] == ["test_a", "test_c", "test_d"]
    config.hook.pytest_deselected.assert_called_once_with(items=removed_expected)

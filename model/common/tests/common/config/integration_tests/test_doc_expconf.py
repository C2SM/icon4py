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
@pytest.mark.snapshots
def test_initial(snap_compare: Callable):
    """Test the initial view."""
    app = config_doc.ConfigDocApp(ExperimentConfig)
    assert snap_compare(app, terminal_size=(160, 40)), (
        "Something has changed the output of the config explorer app. Use `uv run pytest model/common -m snapshots` locally and follow the instructions to visually inspect and accept or fix."
    )


@pytest.mark.level("integration")
@pytest.mark.snapshots
def test_select_one_down(snap_compare: Callable):
    """Test the initial view."""
    app = config_doc.ConfigDocApp(ExperimentConfig)
    assert snap_compare(app, terminal_size=(160, 40), press=["down", "enter"]), (
        "Something has changed the output of the config explorer app. Use `uv run pytest model/common -m snapshots` locally and follow the instructions to visually inspect and accept or fix."
    )

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
import textual.pilot
import textual.widgets

from icon4py.model.common.config import config_doc
from icon4py.model.driver.config import ExperimentConfig


async def _wait_for_loading(pilot: textual.pilot.Pilot) -> None:
    """The tree is filled by a background worker; snapshot only once it is done."""
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.level("integration")
def test_initial(snap_compare: Callable):
    """Test the initial view."""
    app = config_doc.ConfigDocApp(ExperimentConfig)
    assert snap_compare(app, terminal_size=(160, 40), run_before=_wait_for_loading)


@pytest.mark.level("integration")
def test_select_one_down(snap_compare: Callable):
    """Test the initial view."""
    app = config_doc.ConfigDocApp(ExperimentConfig)
    assert snap_compare(
        app, terminal_size=(160, 40), press=["down", "enter"], run_before=_wait_for_loading
    )

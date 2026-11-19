# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest


@pytest.fixture
def snapshot_fail_msg() -> str:
    return "Something has changed the output of the config explorer app, visual inspection required. See summary section."


def pytest_terminal_summary(terminalreporter):
    if any(bool("snapshots" in rep.keywords) for rep in terminalreporter.getreports("failed")):
        terminalreporter.section(
            title="Config Doc App Snapshots Failed - Visual Inspection Required"
        )
        terminalreporter.line("Something has changed the output of the config explorer app.")
        terminalreporter.line(
            "This can happen and is expected when configuration options are added or modified."
        )
        terminalreporter.line(
            "A visual inspection is required to ensure it's not a breaking change."
        )
        terminalreporter.line("")
        terminalreporter.line("If this happened on CI, do the following locally:")
        terminalreporter.line("")
        terminalreporter.line("$ uv run pytest model/common -m snapshots")
        terminalreporter.line("")
        terminalreporter.line(
            "Then follow the instructions in the snapshot summary to visually inspect and accept or fix the changes."
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

from icon4py.liskov.parser import (
    DirectivesInput,
    DirectivesParser,
    IntegrationClassInput,
    IntegrationClassParser,
)


class ExternalInputs:
    """With this class we can specify a configurable list of external inputs to icon liskov."""

    def __init__(self, filepath: Path):
        self.directives = DirectivesParser(filepath)
        self.class_conf = IntegrationClassParser()

    def fetch(self) -> tuple[DirectivesInput, IntegrationClassInput]:
        # todo: create IntegrationInfo
        return self.directives(), self.class_conf()

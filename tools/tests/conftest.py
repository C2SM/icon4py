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

import os
from importlib import reload

import icon4py.model.common.type_alias as type_alias
import pytest
from click.testing import CliRunner


@pytest.fixture
def cli():
    yield CliRunner()
    os.environ["FLOAT_PRECISION"] = type_alias.DEFAULT_PRECISION
    reload(type_alias)

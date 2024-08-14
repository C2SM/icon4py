# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

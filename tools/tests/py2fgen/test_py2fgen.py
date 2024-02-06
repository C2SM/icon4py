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
import pytest
from click.testing import CliRunner

from icon4pytools.py2fgen.cli import main


@pytest.mark.skip(reason="This is skipped. TODO: mixed-precision needs to be fixed in the CI.")
def test_py2fgen():
    cli = CliRunner()
    module = "icon4pytools.py2fgen.wrappers.diffusion_wrapper"
    build_path = "./build"
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module, build_path])
        assert result.exit_code == 0

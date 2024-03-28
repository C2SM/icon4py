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

from icon4pytools.py2fgen.parsing import parse
from icon4pytools.py2fgen.template import CffiPlugin


def test_parse_functions_on_wrapper():
    module_path = "icon4pytools.py2fgen.wrappers.diffusion"
    functions = ["diffusion_init", "diffusion_run"]
    plugin = parse(module_path, functions, "diffusion_plugin")
    assert isinstance(plugin, CffiPlugin)

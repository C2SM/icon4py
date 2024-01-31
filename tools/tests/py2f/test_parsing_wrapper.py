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

from icon4pytools.py2f.cffi_utils import CffiMethod
from icon4pytools.py2f.codegen import CffiPlugin
from icon4pytools.py2f.parsing import parse_function


def test_parse_functions_on_wrapper():
    module_path = "icon4pytools.py2f.wrappers.diffusion_wrapper"
    function_name = "diffusion_init"
    plugin = parse_function(module_path, function_name)
    assert isinstance(plugin, CffiPlugin)
    assert plugin.name == "diffusion_wrapper"


def test_parse_functions_on_program():
    module_path = "icon4py.model.atmosphere.dycore.compute_airmass"
    function_name = "compute_airmass"
    plugin = parse_function(module_path, function_name)
    assert isinstance(plugin, CffiPlugin)
    assert plugin.name == "compute_airmass"


@CffiMethod.register
def do_foo(foo: str):
    return foo


@CffiMethod.register
def do_bar():
    return "bar"


def test_register_with_cffi():
    assert "do_foo" in CffiMethod.get(__name__)
    assert "do_bar" in CffiMethod.get(__name__)

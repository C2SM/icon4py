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

from icon4pytools.py2f.cffi_utils import CffiMethod
from icon4pytools.py2f.parsing import parse_functions_from_module


@pytest.mark.skip
def test_parse_functions():
    path = "icon4pytools.py2f.wrappers.diffusion_wrapper"
    plugin = parse_functions_from_module(path)

    assert plugin.name == "diffusion_wrapper"
    assert len(plugin.functions) == 2
    assert "diffusion_init" in map(lambda f: f.name, plugin.functions)
    assert "diffusion_run" in map(lambda f: f.name, plugin.functions)


@CffiMethod.register
def do_foo(foo: str):
    return foo


@CffiMethod.register
def do_bar():
    return "bar"


def test_register_with_cffi():
    assert "do_foo" in CffiMethod.get(__name__)
    assert "do_bar" in CffiMethod.get(__name__)

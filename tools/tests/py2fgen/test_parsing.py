# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.tools.py2fgen.parsing import parse
from icon4py.tools.py2fgen.template import CffiPlugin


source = """
import foo
import bar

def test_function(x: gtx.Field[gtx.Dims[EdgeDim, KDim], float64], y: int):
    return x * y
"""


def test_parse_functions_on_wrapper():
    module_path = "icon4py.tools.py2fgen.wrappers.diffusion_wrapper"
    functions = ["diffusion_init", "diffusion_run"]
    plugin = parse(module_path, functions, "diffusion_plugin")
    assert isinstance(plugin, CffiPlugin)

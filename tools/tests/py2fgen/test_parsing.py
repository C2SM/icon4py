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

import ast

import pytest

from icon4pytools.py2fgen.parsing import ImportStmtVisitor, TypeHintVisitor, parse
from icon4pytools.py2fgen.template import CffiPlugin


source = """
import foo
import bar

def test_function(x: Field[[EdgeDim, KDim], float64], y: int):
    return x * y
"""


def test_parse_functions_on_wrapper():
    module_path = "icon4pytools.py2fgen.wrappers.diffusion"
    functions = ["diffusion_init", "diffusion_run"]
    plugin = parse(module_path, functions, "diffusion_plugin")
    assert isinstance(plugin, CffiPlugin)


def test_import_visitor():
    tree = ast.parse(source)
    extractor = ImportStmtVisitor()
    extractor.visit(tree)
    expected_imports = ["import foo", "import bar"]
    assert extractor.import_statements == expected_imports


def test_type_hint_visitor():
    tree = ast.parse(source)
    visitor = TypeHintVisitor()
    visitor.visit(tree)
    expected_type_hints = {"x": "Field[[EdgeDim, KDim], float64]", "y": "int"}
    assert visitor.type_hints == expected_type_hints


def test_function_missing_type_hints():
    tree = ast.parse(source.replace(": int", ""))
    visitor = TypeHintVisitor()
    with pytest.raises(TypeError):
        visitor.visit(tree)

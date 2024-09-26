# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast

import pytest

from icon4pytools.py2fgen.parsing import ImportStmtVisitor, TypeHintVisitor, parse
from icon4pytools.py2fgen.template import CffiPlugin


source = """
import foo
import bar

def test_function(x: gtx.Field[gtx.Dims[EdgeDim, KDim], float64], y: int):
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

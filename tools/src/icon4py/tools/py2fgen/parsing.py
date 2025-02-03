# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import importlib
import inspect
from inspect import signature, unwrap
from types import ModuleType
from typing import Callable, List

from gt4py.next.type_system import type_translation as gtx_type_translation

from icon4py.tools.py2fgen.template import CffiPlugin, Func, FuncParameter
from icon4py.tools.py2fgen.utils import parse_type_spec


class ImportStmtVisitor(ast.NodeVisitor):
    """AST Visitor to extract import statements."""

    def __init__(self):
        self.import_statements: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            import_statement = f"import {alias.name}" + (
                f" as {alias.asname}" if alias.asname else ""
            )
            self.import_statements.append(import_statement)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            import_statement = f"from {node.module} import {alias.name}" + (
                f" as {alias.asname}" if alias.asname else ""
            )
            self.import_statements.append(import_statement)


def parse(module_name: str, functions: list[str], plugin_name: str) -> CffiPlugin:
    module = importlib.import_module(module_name)
    parsed_imports = _extract_import_statements(module)
    parsed_functions = [_parse_function(module, f) for f in functions]

    return CffiPlugin(
        module_name=module_name,
        plugin_name=plugin_name,
        functions=parsed_functions,
        imports=parsed_imports,
    )


def _extract_import_statements(module: ModuleType) -> list[str]:
    src = inspect.getsource(module)
    tree = ast.parse(src)
    visitor = ImportStmtVisitor()
    visitor.visit(tree)
    return visitor.import_statements


def _parse_function(module: ModuleType, function_name: str) -> Func:
    func = unwrap(getattr(module, function_name))
    params = _parse_params(func)
    return Func(name=function_name, args=params)


def _parse_params(func: Callable) -> List[FuncParameter]:
    sig_params = signature(func, follow_wrapped=False).parameters
    params = []
    for s, param in sig_params.items():
        gt4py_type = gtx_type_translation.from_type_hint(param.annotation)
        dims, dtype = parse_type_spec(gt4py_type)
        params.append(FuncParameter(name=s, d_type=dtype, dimensions=dims))

    return params

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
import importlib
import inspect
import re
from inspect import signature, unwrap
from typing import Callable

from gt4py.next import Dimension
from gt4py.next.ffront.decorator import Program
from gt4py.next.type_system.type_specifications import ScalarKind
from gt4py.next.type_system.type_translation import from_type_hint

from icon4pytools.py2fgen.codegen import CffiPlugin, Func, FuncParameter
from icon4pytools.py2fgen.common import ARRAY_SIZE_ARGS, parse_type_spec


def parse_function(module_name: str, function_name: str) -> CffiPlugin:
    module = importlib.import_module(module_name)
    parsed_function = _parse_function(module, function_name)
    parsed_imports = _extract_import_statements(module)
    return CffiPlugin(module_name=module_name, function=parsed_function, imports=parsed_imports)


def _parse_function(module, function_name):
    func = unwrap(getattr(module, function_name))
    is_gt4py_program = isinstance(func, Program)
    type_hints = extract_type_hint_strings(module, func, is_gt4py_program, function_name)

    if is_gt4py_program:
        params = _get_gt4py_func_params(func, type_hints)
    else:
        params = _get_simple_func_params(module, func, type_hints)

    return Func(name=function_name, args=params)


def _add_array_size_params(func_params):
    size_param_names = {
        ARRAY_SIZE_ARGS[dim.value]
        for param in func_params
        for dim in param.dimensions
        if dim.value in ARRAY_SIZE_ARGS
    }

    size_params = [
        FuncParameter(name=s, d_type=ScalarKind.INT32, dimensions=[]) for s in size_param_names
    ]

    return func_params + size_params


def _get_gt4py_func_params(func: Program, type_hints) -> list[FuncParameter]:
    """Parse a gt4py program and return its parameters."""
    params = []
    for p in func.past_node.params:
        dims, dtype = parse_type_spec(p.type)
        params.append(
            FuncParameter(name=p.id, d_type=dtype, dimensions=dims, py_type_hint=type_hints[p.id])
        )
    return params


def _get_simple_func_params(module, func: Callable, type_hints) -> list[FuncParameter]:
    """Parse a non-gt4py function and return its parameters."""
    sig_params = signature(func, follow_wrapped=False).parameters

    params = []
    for s in sig_params:
        param = sig_params[s]
        annotation = param.annotation
        type_spec = from_type_hint(annotation)
        dims, dtype = parse_type_spec(type_spec)
        dim_types = [Dimension(value=d.value) for d in dims]
        py_type_hint = type_hints.get(s, None)
        params.append(
            FuncParameter(name=s, d_type=dtype, dimensions=dim_types, py_type_hint=py_type_hint)
        )

    return params


def extract_type_hint_strings(module, func, is_gt4py_program: bool, function_name):
    if is_gt4py_program:
        tmp_src = inspect.getsource(module)
        src = extract_function_signature(tmp_src, function_name)
    else:
        src = inspect.getsource(func)

    tree = ast.parse(src)

    type_hints = {}

    # Define a visitor class to visit function definitions and get type hints
    class FuncDefVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for arg in node.args.args:
                if arg.annotation:
                    annotation = ast.unparse(arg.annotation)
                    type_hints[arg.arg] = annotation

    visitor = FuncDefVisitor()
    visitor.visit(tree)

    return type_hints


def extract_function_signature(code, function_name):
    # Regular expression pattern for a Python function signature
    # This pattern attempts to match function definitions with various parameter types and return annotations
    pattern = (
        rf"\bdef\s+{re.escape(function_name)}\s*\(([\s\S]*?)\)\s*(->\s*[\s\S]*?)?:(?=\s*\n\s*\S)"
    )

    match = re.search(pattern, code)

    if match:
        # Constructing the full signature with return type if it exists
        signature = f"def {function_name}({match.group(1)})"
        if match.group(2):
            signature += f" {match.group(2)}"
        return signature.strip() + ":\n  return None"
    else:
        return "Function signature not found."


def _extract_import_statements(module):
    src = inspect.getsource(module)
    tree = ast.parse(src)

    import_statements = []

    # Define a visitor class to visit import nodes
    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                import_statements.append(
                    f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                )

        def visit_ImportFrom(self, node):
            for alias in node.names:
                import_statements.append(
                    f"from {node.module} import {alias.name}"
                    + (f" as {alias.asname}" if alias.asname else "")
                )

    visitor = ImportVisitor()
    visitor.visit(tree)

    return import_statements

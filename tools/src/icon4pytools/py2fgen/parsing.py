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
    func = _parse_function(module, function_name)
    plugin_name = f"{module_name.split('.')[-1]}_plugin"
    # todo(samkellerhals): for now we just support one function at a time.
    return CffiPlugin(name=plugin_name, function=func)


def _parse_function(module, function_name):
    func = unwrap(getattr(module, function_name))
    if isinstance(func, Program):
        params = _get_gt4py_func_params(func)
        # TODO(samkellerhals): params_with_sizes = _add_array_size_params(params)
        raise Exception(
            "Creating a CffiPlugin for Gt4Py programs without a wrapper is not yet supported."
        )
        # TODO(samkellerhals): Set flag to instruct codegen to generate a wrapper function as we cannot
        #  embed Gt4Py programs directly.
    else:
        # assumes that the simple func implements unpacking of any arrays.
        params = _get_simple_func_params(func)
        # TODO(samkellerhals): params_with_sizes = _add_array_size_params(params)
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


def _get_gt4py_func_params(func: Program) -> list[FuncParameter]:
    """Parse a gt4py program and return its parameters."""
    params = []
    for p in func.past_node.params:
        dims, dtype = parse_type_spec(p.type)
        params.append(FuncParameter(name=p.id, d_type=dtype, dimensions=dims))
    return params


def _get_simple_func_params(func: Callable) -> list[FuncParameter]:
    """Parse a non-gt4py function and return its parameters."""
    sig_params = signature(func, follow_wrapped=False).parameters
    type_hints = extract_type_hint_strings(func)

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


def extract_type_hint_strings(func):
    # Get the source code of the function
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

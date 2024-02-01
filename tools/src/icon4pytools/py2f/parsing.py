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

import importlib
from inspect import signature, unwrap
from typing import Callable

from gt4py.next import Dimension
from gt4py.next.ffront.decorator import Program
from gt4py.next.type_system.type_translation import from_type_hint

from icon4pytools.py2f.codegen import CffiPlugin, Func, FuncParameter
from icon4pytools.py2f.typing_utils import parse_type_spec


def parse_function(module_name: str, function_name: str) -> CffiPlugin:
    module = importlib.import_module(module_name)
    func = _parse_function(module, function_name)
    plugin_name = f"{module_name.split('.')[-1]}_plugin"
    # todo(samkellerhals): for now we just support one function at a time.
    #   as it is not yet clear how to do this with gt4py programs
    return CffiPlugin(name=plugin_name, functions=[func])


def _parse_function(module, function_name):
    func = unwrap(getattr(module, function_name))
    if isinstance(func, Program):
        return _parse_gt4py_program(func, function_name)
    else:
        return _parse_non_gt4py_function(func, function_name)


def _parse_gt4py_program(func: Program, function_name: str) -> Func:
    args = []
    for p in func.past_node.params:
        dims, dtype = parse_type_spec(p.type)
        args.append(FuncParameter(name=p.id, d_type=dtype, dimensions=dims))
    return Func(name=function_name, args=args)


def _parse_non_gt4py_function(func: Callable, function_name: str) -> Func:
    args = [
        _parse_params(signature(func, follow_wrapped=False).parameters, p)
        for p in (signature(func).parameters)
    ]
    return Func(name=function_name, args=args)


def _parse_params(params, s):
    annotation = params[s].annotation
    type_spec = from_type_hint(annotation)
    dims, dtype = parse_type_spec(type_spec)
    dim_types = [Dimension(value=d.value) for d in dims]  # todo: why is this always 10?
    return FuncParameter(name=s, d_type=dtype, dimensions=dim_types)

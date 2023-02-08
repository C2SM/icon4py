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

from functional.type_system.type_specifications import ScalarType
from functional.type_system.type_translation import from_type_hint

from icon4py.py2f.cffi_utils import CffiMethod
from icon4py.py2f.codegen import CffiPlugin, DimensionType, Func, FuncParameter


def parse_functions_from_module(module_name: str) -> CffiPlugin:
    module = importlib.import_module(module_name)
    func_names = CffiMethod.get(module_name)
    funcs = [_parse_function(module, fn) for fn in func_names]
    plugin_name = module_name.split(".")[-1]
    return CffiPlugin(name=plugin_name, functions=funcs)


def _parse_function(module, s):
    func = unwrap(getattr(module, s))
    params = [
        _parse_params(signature(func, follow_wrapped=False).parameters, p)
        for p in (signature(func).parameters)
    ]
    return Func(name=s, args=params)


def _parse_params(params, s):
    type_spec = from_type_hint(params[s].annotation)
    if isinstance(type_spec, ScalarType):
        dtype = type_spec.kind
        dims = []
    else:
        dtype = type_spec.dtype.kind
        dims = [DimensionType(name=d.value, length=10) for d in type_spec.dims]
    return FuncParameter(name=s, d_type=dtype, dimensions=dims)

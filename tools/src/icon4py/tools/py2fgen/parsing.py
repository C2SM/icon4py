# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from typing import Callable

from icon4py.tools.py2fgen import _template


def get_cffi_description(
    module_name: str, functions: list[str], plugin_name: str
) -> _template.CffiPlugin:
    module = importlib.import_module(module_name)
    parsed_functions = [_get_function_descriptor(getattr(module, f)) for f in functions]
    return _template.CffiPlugin(
        module_name=module_name,
        plugin_name=plugin_name,
        functions=parsed_functions,
    )


def _get_function_descriptor(fun: Callable) -> _template.Func:
    if not hasattr(fun, "function_descriptor"):
        raise TypeError("Cannot parse function, did you forget to decorate it with '@export'?")
    return fun.function_descriptor

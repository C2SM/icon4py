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

from typing import Optional

from gt4py.eve import codegen

from icon4pytools.icon4pygen.bindings.utils import format_fortran_code
from icon4pytools.py2fgen.template import (
    CffiPlugin,
    CHeaderGenerator,
    F90InterfaceGenerator,
    PythonWrapper,
    PythonWrapperGenerator,
)


def generate_c_header(plugin: CffiPlugin) -> str:
    """
    Generate C header code from the given plugin.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.

    Returns:
        Formatted C header code as a string.
    """
    generated_code = CHeaderGenerator.apply(plugin)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def generate_python_wrapper(
    plugin: CffiPlugin, gt4py_backend: Optional[str], debug_mode: bool
) -> str:
    """
    Generate Python wrapper code.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.
        gt4py_backend: Optional gt4py backend specification.
        debug_mode: Flag indicating if debug mode is enabled.

    Returns:
        Formatted Python wrapper code as a string.
    """
    python_wrapper = PythonWrapper(
        module_name=plugin.module_name,
        function=plugin.function,
        imports=plugin.imports,
        gt4py_backend=gt4py_backend,
        debug_mode=debug_mode,
    )
    generated_code = PythonWrapperGenerator.apply(python_wrapper)
    return codegen.format_source("python", generated_code)


def generate_f90_interface(plugin: CffiPlugin) -> str:
    """
    Generate Fortran 90 interface code.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.
    """
    generated_code = F90InterfaceGenerator.apply(plugin)
    return format_fortran_code(generated_code)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from gt4py.eve import codegen

from icon4pytools.common.logger import setup_logger
from icon4pytools.common.utils import format_fortran_code
from icon4pytools.py2fgen.template import (
    CffiPlugin,
    CHeaderGenerator,
    F90Interface,
    F90InterfaceGenerator,
    PythonWrapper,
    PythonWrapperGenerator,
)


logger = setup_logger(__name__)


def generate_c_header(plugin: CffiPlugin) -> str:
    """
    Generate C header code from the given plugin.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.

    Returns:
        Formatted C header code as a string.
    """
    logger.info("Generating C header...")
    generated_code = CHeaderGenerator.apply(plugin)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def generate_python_wrapper(
    plugin: CffiPlugin, backend: Optional[str], debug_mode: bool, limited_area: str, profile: bool
) -> str:
    """
    Generate Python wrapper code.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.
        backend: Optional gt4py backend specification.
        debug_mode: Flag indicating if debug mode is enabled.
        limited_area: Optional gt4py limited area specification.
        profile: Flag indicate if code should be profiled.

    Returns:
        Formatted Python wrapper code as a string.
    """
    logger.info("Generating Python wrapper...")
    python_wrapper = PythonWrapper(
        module_name=plugin.module_name,
        plugin_name=plugin.plugin_name,
        functions=plugin.functions,
        imports=plugin.imports,
        backend=backend,
        debug_mode=debug_mode,
        limited_area=limited_area,
        profile=profile,
    )

    generated_code = PythonWrapperGenerator.apply(python_wrapper)
    return codegen.format_source("python", generated_code)


def generate_f90_interface(plugin: CffiPlugin, limited_area: str) -> str:
    """
    Generate Fortran 90 interface code.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.
    """
    logger.info("Generating Fortran interface...")
    generated_code = F90InterfaceGenerator.apply(
        F90Interface(cffi_plugin=plugin, limited_area=limited_area)
    )
    return format_fortran_code(generated_code)

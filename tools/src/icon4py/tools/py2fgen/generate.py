# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.eve import codegen

from icon4py.tools.common.logger import setup_logger
from icon4py.tools.common.utils import format_fortran_code
from icon4py.tools.py2fgen._template import (
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


def generate_python_wrapper(plugin: CffiPlugin) -> str:
    """
    Generate Python wrapper code.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.

    Returns:
        Formatted Python wrapper code as a string.
    """
    logger.info("Generating Python wrapper...")
    python_wrapper = PythonWrapper(
        module_name=plugin.module_name,
        plugin_name=plugin.plugin_name,
        functions=plugin.functions,
    )

    generated_code = PythonWrapperGenerator.apply(python_wrapper)
    return codegen.format_source("python", generated_code)


def generate_f90_interface(plugin: CffiPlugin) -> str:
    """
    Generate Fortran 90 interface code.

    Args:
        plugin: The CffiPlugin instance containing information for code generation.
    """
    logger.info("Generating Fortran interface...")
    generated_code = F90InterfaceGenerator.apply(F90Interface(cffi_plugin=plugin))
    return format_fortran_code(generated_code)

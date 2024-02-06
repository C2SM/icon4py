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
from pathlib import Path
from typing import Optional, Sequence

from gt4py.eve import Node, codegen
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator
from gt4py.next import Dimension
from gt4py.next.type_system.type_specifications import ScalarKind

from icon4pytools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
)
from icon4pytools.icon4pygen.bindings.utils import write_string, format_fortran_code
from icon4pytools.py2fgen.common import ARRAY_SIZE_ARGS


class DimensionType(Node):
    name: str
    length: int


class FuncParameter(Node):
    name: str
    d_type: ScalarKind
    dimensions: Sequence[Optional[Dimension]]


class Func(Node):
    name: str
    args: Sequence[FuncParameter]


class CffiPlugin(Node):
    name: str
    functions: Sequence[Func]


def to_c_type(scalar_type: ScalarKind) -> str:
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_f_type(scalar_type: ScalarKind) -> str:
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: FuncParameter) -> str:
    """
    If param is a scalar type (dimension=0) then return the F90 'value' keyword.

    Used for F90 generation only.
    """
    return "value," if len(param.dimensions) == 0 else ""


def get_intent(param: FuncParameter) -> str:
    # todo(samkellerhals): quick hack to set correct intent for domain bounds
    #   by default all other variables are assumed to be arrays for now
    #   and passed by reference (pointers) in the C interface and thus
    #   annotated with inout in the f90 interface. All params need to have
    #   corresponding in/out/inout types associated with them going forward
        if "_start" in param.name or "_end" in param.name:
            return "in"
        else:
            return "inout"


def as_field(param: FuncParameter, language: str) -> str:
    size = len(param.dimensions)
    if size == 0:
        return ""
    # TODO(samkellerhals): refactor into separate function
    if "C" == language:
        return "" if len(param.dimensions) == 0 else "*"
    else:
        dims = ",".join(map(lambda x: ":", range(size)))
        return f"dimension({dims}),"

def get_array_size_params(param: FuncParameter) -> str:
    size = len(param.dimensions)
    if size == 0:
        return ""

    # Map the dimension values to their corresponding Fortran array access strings
    access_strings = []
    for dim in param.dimensions:
        if dim.value in ARRAY_SIZE_ARGS:
            fortran_dim = ARRAY_SIZE_ARGS[dim.value]
            access_strings.append(fortran_dim)

    # Format the array access string for Fortran
    return "(" + ", ".join(access_strings) + ")"


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""{% for func in functions: %}{{func}}\n{% endfor %}""")

    Func = as_jinja("""extern void {{name}}({{", ".join(args)}});""")

    def visit_FuncParameter(self, param: FuncParameter):
        return self.generic_visit(
            param,
            rendered_type=to_c_type(param.d_type),
            dim=as_field(param, "C"),
        )

    FuncParameter = as_jinja("""{{rendered_type}}{{dim}} {{name}}""")


# todo(samkellerhals): code needs to be formatted
class F90InterfaceGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja(
        """\
    module {{name}}
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
    {% for func in functions: %}\
    {{func}}\
    {% endfor %}\
    end interface
    end module
    """
    )

    def visit_Func(self, func: Func):
        arg_names = ", &\n ".join(map(lambda x: x.name, func.args))
        return self.generic_visit(func, param_names=arg_names)

    Func = as_jinja(
        """subroutine {{name}}({{param_names}}) bind(c, name='{{name}}')
       use, intrinsic :: iso_c_binding
       {% for arg in args: %}\
       {{arg}}\
       {% endfor %}\
    end subroutine {{name}}
    """
    )

    FuncParameter = as_jinja(
        """{{rendered_type}}, {{dim}} {{value}} target :: {{name}}{{ explicit_size }}
    """
    )

    def visit_FuncParameter(self, param: FuncParameter, param_names=""):
        # kw-arg param_names needs to be present because it is present up the tree
        return self.generic_visit(
            param,
            value=as_f90_value(param),
            intent=get_intent(param),
            rendered_type=to_f_type(param.d_type),
            dim=as_field(param, "F"),
            explicit_size=get_array_size_params(param)
        )


def generate_c_header(plugin: CffiPlugin) -> str:
    generated_code = CHeaderGenerator.apply(plugin)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def generate_and_write_f90_interface(build_path: Path, plugin: CffiPlugin):
    generated_code = F90InterfaceGenerator.apply(plugin)
    formatted_source = format_fortran_code(generated_code)
    write_string(formatted_source, build_path, f"{plugin.name}.f90")

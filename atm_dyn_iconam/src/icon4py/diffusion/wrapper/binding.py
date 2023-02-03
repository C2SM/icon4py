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

from typing import Sequence

import eve
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator
from functional.type_system.type_specifications import ScalarKind

from icon4py.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
)


class DimensionType(eve.Node):
    name: str
    length: int


class FuncParameter(eve.Node):
    name: str
    d_type: ScalarKind
    dimensions: Sequence[DimensionType]


class Func(eve.Node):
    name: str
    args: Sequence[FuncParameter]


class CffiPlugin(eve.Node):
    name: str
    functions: Sequence[Func]


def to_c_type(scalar_type: ScalarKind):
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_f_type(scalar_type: ScalarKind):
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def field_extension(param: FuncParameter, language: str) -> str:
    size = len(param.dimensions)
    if size == 0:
        return ""
    if "C" == language:
        return "*"
    else:
        dims = ",".join(map(lambda x: ":", range(size)))
        return f"({dims})"


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""{% for func in functions: %}{{func}}\n{% endfor %}""")

    Func = as_jinja("""extern void {{name}}({{", ".join(args)}});""")

    def visit_FuncParameter(self, param: FuncParameter):
        return self.generic_visit(
            param,
            rendered_type=to_c_type(param.d_type),
            dim=field_extension(param, "C"),
        )

    FuncParameter = as_jinja("""{{rendered_type}}{{dim}} {{name}}""")


class FortranInterfaceGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja(
        """
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
        arg_names = ", ".join(map(lambda x: x.name, func.args))
        return self.generic_visit(func, param_names=arg_names)

    Func = as_jinja(
        """subroutine {{name}}({{param_names}}) bind(c, name='{{name}}')
        use iso_c_binding
       {% for arg in args: %}\
       {{arg}}\
       {% endfor %}\
    end subroutine {{name}}
    """
    )

    def visit_FuncParameter(self, param: FuncParameter, param_names=""):
        return self.generic_visit(param, rendered_type=to_f_type(param.d_type))

    FuncParameter = as_jinja(
        """{{rendered_type}}, intent(inout):: {{name}}
    """
    )

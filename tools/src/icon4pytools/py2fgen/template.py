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
from typing import Any, Optional, Sequence

from gt4py.eve import Node, datamodels
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator
from gt4py.next import Dimension
from gt4py.next.type_system.type_specifications import ScalarKind

from icon4pytools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
)
from icon4pytools.py2fgen.utils import (
    ARRAY_SIZE_ARGS,
    CFFI_DECORATOR,
    CFFI_UNPACK,
    flatten_and_get_unique_elts,
)


class DimensionType(Node):
    name: str
    length: int


class FuncParameter(Node):
    name: str
    d_type: ScalarKind
    dimensions: Sequence[Dimension]
    py_type_hint: str
    size_args: list[str] = datamodels.field(init=False)
    is_array: bool = datamodels.field(init=False)
    gtdims: list[str] = datamodels.field(init=False)

    def __post_init__(self):
        self.size_args = dims_to_size_strings(self.dimensions)
        self.is_array = True if len(self.dimensions) >= 1 else False
        self.gtdims = [dimension.value + "Dim" for dimension in self.dimensions]


class Func(Node):
    name: str
    args: Sequence[FuncParameter]
    is_gt4py_program: bool
    size_args: Sequence[str] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.size_args = flatten_and_get_unique_elts(
            [dims_to_size_strings(arg.dimensions) for arg in self.args]
        )


class CffiPlugin(Node):
    module_name: str
    function: Func
    imports: list[str]
    plugin_name: str = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.plugin_name = f"{self.module_name.split('.')[-1]}_plugin"


def to_c_type(scalar_type: ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_f_type(scalar_type: ScalarKind) -> str:
    """Convert a scalar type to its corresponding ISO C type."""
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: FuncParameter) -> str:
    """
    Return the Fortran 90 'value' keyword for scalar types.

    Args:
        param: The function parameter to check.

    Returns:
        A string containing 'value,' for scalar types, otherwise an empty string.
    """
    return "value," if len(param.dimensions) == 0 else ""


def render_c_pointer(param: FuncParameter) -> str:
    """Render a C pointer symbol for array types."""
    return "*" if len(param.dimensions) > 0 else ""


def render_fortran_array_dimensions(param: FuncParameter) -> str:
    """
    Render Fortran array dimensions for array types.

    Args:
        param: The function parameter to check.

    Returns:
        A string representing Fortran array dimensions.
    """
    if len(param.dimensions) > 0:
        dims = ",".join(":" for _ in param.dimensions)
        return f"dimension({dims}),"
    return ""


# TODO(samkellerhals): We should throw an exception if the dimension is not found in our ARRAY_SIZE_ARGS list
def dims_to_size_strings(dimensions: Sequence[Dimension]) -> list[str]:
    """
    Convert dimension values to Fortran array access strings.

    Args:
        dimensions: A sequence of dimensions to convert.

    Returns:
        A list of Fortran array size strings.
    """
    return sorted(ARRAY_SIZE_ARGS[dim.value] for dim in dimensions if dim.value in ARRAY_SIZE_ARGS)


def render_fortran_array_sizes(param: FuncParameter) -> str:
    """
    Render Fortran array size strings for array parameters.

    Args:
        param: The function parameter to check.

    Returns:
        A string representing Fortran array sizes.
    """
    if len(param.dimensions) > 0:
        size_strings = dims_to_size_strings(param.dimensions)
        return "(" + ", ".join(size_strings) + ")"
    return ""


class PythonWrapper(CffiPlugin):
    gt4py_backend: Optional[str]
    debug_mode: bool
    size_args: Sequence[str] = datamodels.field(init=False)
    plugin_name: str = datamodels.field(init=False)
    cffi_decorator: str = CFFI_DECORATOR
    cffi_unpack: str = CFFI_UNPACK

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.size_args = flatten_and_get_unique_elts(
            [dims_to_size_strings(arg.dimensions) for arg in self.function.args]
        )
        self.plugin_name = f"{self.module_name.split('.')[-1]}_plugin"  # TODO(samkellerhals): Consider setting this in the CLI.


class PythonWrapperGenerator(TemplatedGenerator):
    PythonWrapper = as_jinja(
        """\
# necessary imports for generated code to work
from {{ plugin_name }} import ffi
import numpy as np
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_gpu
from gt4py.next.program_processors.runners.roundtrip import backend as run_roundtrip
from icon4py.model.common.grid.simple import SimpleGrid

# all other imports from the module from which the function is being wrapped
{% for stmt in imports -%}
{{ stmt }}
{% endfor %}

# We need a grid to pass offset providers
grid = SimpleGrid()

from {{ module_name }} import {{ _this_node.function.name }}

{{ cffi_unpack }}

{{ cffi_decorator }}
def {{ _this_node.function.name }}_wrapper(
{%- for arg in _this_node.function.args -%}
{{ arg.name }}: {{ arg.py_type_hint }}{% if not loop.last or _this_node.size_args %}, {% endif %}
{%- endfor %}
{%- for arg in _this_node.size_args -%}
{{ arg }}: int32{{ ", " if not loop.last else "" }}
{%- endfor -%}
):

    # Unpack pointers into Ndarrays
    {% for arg in _this_node.function.args %}
    {% if arg.is_array %}
    {{ arg.name }} = unpack({{ arg.name }}, {{ ", ".join(arg.size_args) }})
    {%- if _this_node.debug_mode %}
    print({{ arg.name }})
    print({{ arg.name }}.shape)
    {% endif %}
    {% endif %}
    {% endfor %}

    # Allocate GT4Py Fields
    {% for arg in _this_node.function.args %}
    {% if arg.is_array %}
    {{ arg.name }} = np_as_located_field({{ ", ".join(arg.gtdims) }})({{ arg.name }})
    {%- if _this_node.debug_mode %}
    print({{ arg.name }})
    print({{ arg.name }}.shape)
    {% endif %}
    {% endif %}
    {% endfor %}

    {{ _this_node.function.name }}
    {%- if _this_node.function.is_gt4py_program -%}.with_backend({{ _this_node.gt4py_backend }}){%- endif -%}(
    {%- for arg in _this_node.function.args -%}
    {{ arg.name }}{{ ", " if not loop.last or _this_node.function.is_gt4py_program else "" }}
    {%- endfor -%}
    {%- if _this_node.function.is_gt4py_program -%}
    offset_provider=grid.offset_providers
    {%- endif -%}
)
"""
    )


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""extern void {{_this_node.function.name}}_wrapper({{function}});""")

    Func = as_jinja(
        "{%- for arg in args -%}{{ arg }}{% if not loop.last or size_args|length > 0 %}, {% endif %}{% endfor -%}{%- for sarg in size_args -%} int {{ sarg }}{% if not loop.last %}, {% endif %}{% endfor -%}"
    )

    def visit_FuncParameter(self, param: FuncParameter):
        return self.generic_visit(
            param, rendered_type=to_c_type(param.d_type), pointer=render_c_pointer(param)
        )

    FuncParameter = as_jinja("""{{rendered_type}}{{pointer}} {{name}}""")


class F90InterfaceGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja(
        """\
    module {{ plugin_name }}
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
    {{ function }}
    end interface
    end module
    """
    )

    def visit_Func(self, func: Func, **kwargs):
        arg_names = ", &\n ".join(map(lambda x: x.name, func.args))
        if func.size_args:
            arg_names += ",&\n" + ", &\n".join(func.size_args)

        return self.generic_visit(func, param_names=arg_names)

    Func = as_jinja(
        """subroutine {{name}}_wrapper({{param_names}}) bind(c, name='{{name}}_wrapper')
       use, intrinsic :: iso_c_binding
       {% for size_arg in size_args -%}
       integer(c_int), value, target :: {{ size_arg }}
       {% endfor %}
       {% for arg in args: %}\
       {{arg}}\
       {% endfor %}\
    end subroutine {{name}}_wrapper
    """
    )

    def visit_FuncParameter(self, param: FuncParameter, **kwargs):
        return self.generic_visit(
            param,
            value=as_f90_value(param),
            rendered_type=to_f_type(param.d_type),
            dim=render_fortran_array_dimensions(param),
            explicit_size=render_fortran_array_sizes(param),
        )

    FuncParameter = as_jinja(
        """{{rendered_type}}, {{dim}} {{value}} target :: {{name}}{{ explicit_size }}
    """
    )

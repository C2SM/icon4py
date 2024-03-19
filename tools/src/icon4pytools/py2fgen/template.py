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
import inspect
from typing import Any, Sequence

from gt4py.eve import Node, datamodels
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from gt4py.next import Dimension
from gt4py.next.type_system.type_specifications import ScalarKind

from icon4pytools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
)
from icon4pytools.py2fgen.plugin import int_array_to_bool_array, unpack, unpack_gpu
from icon4pytools.py2fgen.utils import GT4PyBackend, flatten_and_get_unique_elts


CFFI_DECORATOR = "@ffi.def_extern()"
PROGRAM_DECORATOR = "@program"


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
    global_size_args: Sequence[str] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.global_size_args = flatten_and_get_unique_elts(
            [dims_to_size_strings(arg.dimensions) for arg in self.args]
        )


class CffiPlugin(Node):
    module_name: str
    plugin_name: str
    imports: list[str]
    function: list[Func]


class PythonWrapper(CffiPlugin):
    backend: str
    debug_mode: bool
    cffi_decorator: str = CFFI_DECORATOR
    cffi_unpack: str = inspect.getsource(unpack)
    cffi_unpack_gpu: str = inspect.getsource(unpack_gpu)
    int_to_bool: str = inspect.getsource(int_array_to_bool_array)
    gt4py_backend: str = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.gt4py_backend = GT4PyBackend[self.backend].value


def build_array_size_args() -> dict[str, str]:
    array_size_args = {}
    from icon4py.model.common import dimension

    for var_name, var in vars(dimension).items():
        if isinstance(var, Dimension):
            dim_name = var_name.replace(
                "Dim", ""
            )  # Assumes we keep suffixing each Dimension with Dim
            size_name = f"n_{dim_name}"
            array_size_args[dim_name] = size_name
    return array_size_args


def to_c_type(scalar_type: ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_iso_c_type(scalar_type: ScalarKind) -> str:
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


def render_fortran_array_dimensions(param: FuncParameter, assumed_size_array: bool) -> str:
    """
    Render Fortran array dimensions for array types.

    Args:
        param: The function parameter to check.

    Returns:
        A string representing Fortran array dimensions.
    """
    if len(param.dimensions) > 0 and assumed_size_array:
        return "dimension(*),"

    if len(param.dimensions) > 0 and not assumed_size_array:
        dims = ",".join(":" for _ in param.dimensions)
        return f"dimension({dims}),"

    return ""


def dims_to_size_strings(dimensions: Sequence[Dimension]) -> list[str]:
    """Convert Python array dimension values to Fortran array access strings.

    These already should be in Row-major order as defined in the Python function
    definition type hints for each array. These will be used to reshape the
    Column-major ordered arrays from Fortran in the unpack function.

    Args:
        dimensions: A sequence of dimensions to convert.

    Returns:
        A list of Fortran array size strings.
    """
    array_size_args = build_array_size_args()
    size_strings = []
    for dim in dimensions:
        if dim.value in array_size_args:
            size_strings.append(array_size_args[dim.value])
        else:
            raise ValueError(f"Dimension '{dim.value}' not found in ARRAY_SIZE_ARGS")
    return size_strings


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


class PythonWrapperGenerator(TemplatedGenerator):
    PythonWrapper = as_jinja(
        """\
# necessary imports for generated code to work
from {{ plugin_name }} import ffi
import numpy as np
import cupy as cp
from numpy.typing import NDArray
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next import as_field
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_gpu
from gt4py.next.program_processors.runners.roundtrip import backend as run_roundtrip
from icon4py.model.common.grid.simple import SimpleGrid

# all other imports from the module from which the function is being wrapped
{% for stmt in imports -%}
{{ stmt }}
{% endfor %}

import logging

log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'

logging.basicConfig(filename='py_cffi.log',
                    level=logging.DEBUG,
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')

# We need a grid to pass offset providers
grid = SimpleGrid()

{% for func in _this_node.function %}
from {{ module_name }} import {{ func.name }}
{% endfor %}

{{ cffi_unpack }}

{{ cffi_unpack_gpu }}

{{ int_to_bool }}

def list_available_gpus():
    num_gpus = cp.cuda.runtime.getDeviceCount()
    logging.debug('Total GPUs available: %d' % num_gpus)

    for i in range(num_gpus):
        device = cp.cuda.Device(i)
        logging.debug(device)

{% for func in _this_node.function %}

{{ cffi_decorator }}
def {{ func.name }}_wrapper(
{%- for arg in func.args -%}
{{ arg.name }}: {{ arg.py_type_hint }}{% if not loop.last or func.global_size_args %}, {% endif %}
{%- endfor %}
{%- for arg in func.global_size_args -%}
{{ arg }}: int32{{ ", " if not loop.last else "" }}
{%- endfor -%}
):
    try:
        {%- if _this_node.debug_mode %}
        logging.info("Python Execution Context Start")
        {% endif %}

        list_available_gpus()

        # Unpack pointers into Ndarrays
        {% for arg in func.args %}
        {% if arg.is_array %}
        {%- if _this_node.debug_mode %}
        msg = '{{ arg.name }} before unpacking: %s' % str({{ arg.name}})
        logging.debug(msg)
        {% endif %}
        {{ arg.name }} = unpack{%- if _this_node.backend == 'GPU' -%}_gpu{%- endif -%}({{ arg.name }}, {{ ", ".join(arg.size_args) }})

        {%- if arg.d_type.name == "BOOL" %}
        {{ arg.name }} = int_array_to_bool_array({{ arg.name }})
        {%- endif %}

        {%- if _this_node.debug_mode %}
        msg = '{{ arg.name }} after unpacking: %s' % str({{ arg.name}})
        logging.debug(msg)
        msg = 'shape of {{ arg.name }} after unpacking = %s' % str({{ arg.name}}.shape)
        logging.debug(msg)
        {% endif %}
        {% endif %}
        {% endfor %}

        # Allocate GT4Py Fields
        {% for arg in func.args %}
        {% if arg.is_array %}
        {{ arg.name }} = np_as_located_field({{ ", ".join(arg.gtdims) }})({{ arg.name }})
        {%- if _this_node.debug_mode %}
        msg = 'shape of {{ arg.name }} after allocating as field = %s' % str({{ arg.name}}.shape)
        logging.debug(msg)
        msg = '{{ arg.name }} after allocating as field: %s' % str({{ arg.name }}.ndarray)
        logging.debug(msg)
        {% endif %}
        {% endif %}
        {% endfor %}

        {{ func.name }}
        {%- if func.is_gt4py_program -%}.with_backend({{ _this_node.gt4py_backend }}){%- endif -%}(
        {%- for arg in func.args -%}
        {{ arg.name }}{{ ", " if not loop.last or func.is_gt4py_program else "" }}
        {%- endfor -%}
        {%- if func.is_gt4py_program -%}
        offset_provider=grid.offset_providers
        {%- endif -%}
        )

        {% if _this_node.debug_mode %}
        # debug info
        {% for arg in func.args %}
        {% if arg.is_array %}
        msg = 'shape of {{ arg.name }} after computation = %s' % str({{ arg.name}}.shape)
        logging.debug(msg)
        msg = '{{ arg.name }} after computation: %s' % str({{ arg.name }}.ndarray)
        logging.debug(msg)
        {% endif %}
        {% endfor %}
        {% endif %}

        {%- if _this_node.debug_mode %}
        logging.info("Python Execution Context End")
        {% endif %}

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0

        {% endfor %}


"""
    )


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""{{'\n'.join(function)}}""")

    Func = as_jinja(
        "extern int {{ name }}_wrapper({%- for arg in args -%}{{ arg }}{% if not loop.last or global_size_args|length > 0 %}, {% endif %}{% endfor -%}{%- for sarg in global_size_args -%} int {{ sarg }}{% if not loop.last %}, {% endif %}{% endfor -%});"
    )

    def visit_FuncParameter(self, param: FuncParameter):
        return self.generic_visit(
            param, rendered_type=to_c_type(param.d_type), pointer=render_c_pointer(param)
        )

    FuncParameter = as_jinja("""{{rendered_type}}{{pointer}} {{name}}""")


class F90FunctionDeclaration(Func):
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # call Func __post_init__


class DimensionPosition(Node):
    variable: str
    size_arg: str
    index: int


class F90FunctionDefinition(Func):
    dimension_size_declarations: Sequence[DimensionPosition] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # call Func __post_init__

        dim_positions = []
        for arg in self.args:
            for index, size_arg in enumerate(arg.size_args):
                dim_positions.append(
                    DimensionPosition(variable=str(arg.name), size_arg=size_arg, index=index + 1)
                )  # Use Fortran indexing

        self.dimension_size_declarations = dim_positions


class F90Interface(Node):
    cffi_plugin: CffiPlugin
    function_declaration: list[F90FunctionDeclaration] = datamodels.field(init=False)
    function_definition: list[F90FunctionDefinition] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        functions = self.cffi_plugin.function
        self.function_declaration = [
            F90FunctionDeclaration(name=f.name, args=f.args, is_gt4py_program=f.is_gt4py_program)
            for f in functions
        ]
        self.function_definition = [
            F90FunctionDefinition(name=f.name, args=f.args, is_gt4py_program=f.is_gt4py_program)
            for f in functions
        ]


class F90InterfaceGenerator(TemplatedGenerator):
    F90Interface = as_jinja(
        """\
module {{ _this_node.cffi_plugin.plugin_name }}
    use, intrinsic :: iso_c_binding
    implicit none

    {% for func in _this_node.cffi_plugin.function %}
    public :: {{ func.name }}
    {% endfor %}

interface
    {{ '\n'.join(function_declaration) }}
end interface

contains
    {{ '\n'.join(function_definition) }}
end module
"""
    )

    def visit_F90FunctionDeclaration(self, func: F90FunctionDeclaration, **kwargs):
        arg_names = ", &\n ".join(map(lambda x: x.name, func.args))
        if func.global_size_args:
            arg_names += ",&\n" + ", &\n".join(func.global_size_args)
        return self.generic_visit(func, assumed_size_array=True, param_names=arg_names)

    F90FunctionDeclaration = as_jinja(
        """
function {{name}}_wrapper({{param_names}}) bind(c, name="{{name}}_wrapper") result(rc)
   import :: c_int, c_double, c_bool, c_ptr
   {% for size_arg in global_size_args %}
   integer(c_int), value :: {{ size_arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code
   {% for arg in args %}
   {{ arg }}
   {% endfor %}
end function {{name}}_wrapper
    """
    )

    def visit_F90FunctionDefinition(self, func: F90FunctionDefinition, **kwargs):
        if len(func.args) < 1:
            arg_names, param_names_with_size_args = "", ""
        else:
            arg_names = ", &\n ".join(map(lambda x: x.name, func.args))
            param_names_with_size_args = arg_names + ",&\n" + ", &\n".join(func.global_size_args)

        return self.generic_visit(
            func,
            assumed_size_array=False,
            param_names=arg_names,
            param_names_with_size_args=param_names_with_size_args,
        )

    # todo(samkellerhals): Consider using unique SIZE args
    F90FunctionDefinition = as_jinja(
        """
subroutine {{name}}({{param_names}}, &\nrc)
   use, intrinsic :: iso_c_binding
   {% for size_arg in global_size_args %}
   integer(c_int) :: {{ size_arg }}
   {% endfor %}
   {% for arg in args %}
   {{ arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code

   {% for d in _this_node.dimension_size_declarations %}
   {{ d.size_arg }} = SIZE({{ d.variable }}, {{ d.index }})
   {% endfor %}

   rc = {{ name }}_wrapper({{ param_names_with_size_args }})

end subroutine {{name}}
    """
    )

    def visit_FuncParameter(self, param: FuncParameter, **kwargs):
        return self.generic_visit(
            param,
            value=as_f90_value(param),
            iso_c_type=to_iso_c_type(param.d_type),
            dim=render_fortran_array_dimensions(param, kwargs["assumed_size_array"]),
            explicit_size=render_fortran_array_sizes(param),
        )

    FuncParameter = as_jinja("""{{iso_c_type}}, value :: {{name}}""")

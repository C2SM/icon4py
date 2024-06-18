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
from icon4py.model.common.config import GT4PyBackend

from icon4pytools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
    BUILTIN_TO_NUMPY_TYPE,
)
from icon4pytools.py2fgen.plugin import int_array_to_bool_array, unpack, unpack_gpu
from icon4pytools.py2fgen.utils import flatten_and_get_unique_elts
from icon4pytools.py2fgen.wrappers.experiments import UNINITIALISED_ARRAYS


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
    size_args_len: int = datamodels.field(init=False)
    np_type: str = datamodels.field(init=False)

    def __post_init__(self):
        self.size_args = dims_to_size_strings(self.dimensions)
        self.size_args_len = len(self.size_args)
        self.is_array = True if len(self.dimensions) >= 1 else False
        # We need some fields to have nlevp1 levels on the fortran wrapper side, which we make
        # happen by using KHalfDim as a type hint. However, this is not yet supported on the icon4py
        # side. So before generating the python wrapper code, we replace occurrences of KHalfDim with KDim
        self.gtdims = [
            dimension.value.replace("KHalf", "K") + "Dim" for dimension in self.dimensions
        ]
        self.np_type = to_np_type(self.d_type)


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
    functions: list[Func]


class PythonWrapper(CffiPlugin):
    backend: str
    debug_mode: bool
    profile: bool
    limited_area: bool
    cffi_decorator: str = CFFI_DECORATOR
    cffi_unpack: str = inspect.getsource(unpack)
    cffi_unpack_gpu: str = inspect.getsource(unpack_gpu)
    int_to_bool: str = inspect.getsource(int_array_to_bool_array)
    gt4py_backend: str = datamodels.field(init=False)
    is_gt4py_program_present: bool = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.gt4py_backend = GT4PyBackend[self.backend].value
        self.is_gt4py_program_present = any(func.is_gt4py_program for func in self.functions)
        self.uninitialised_arrays = get_uninitialised_arrays(self.limited_area)


def get_uninitialised_arrays(limited_area: bool):
    return UNINITIALISED_ARRAYS if not limited_area else []


def build_array_size_args() -> dict[str, str]:
    array_size_args = {}
    from icon4py.model.common import dimension

    for var_name, var in vars(dimension).items():
        if isinstance(var, Dimension):
            dim_name = var_name.replace(
                "Dim", ""
            )  # Assumes we keep suffixing each Dimension with Dim in icon4py.common.dimension module
            size_name = f"n_{dim_name}"
            array_size_args[dim_name] = size_name
    return array_size_args


def to_c_type(scalar_type: ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_np_type(scalar_type: ScalarKind) -> str:
    """Convert a scalar type to its corresponding numpy type."""
    return BUILTIN_TO_NUMPY_TYPE[scalar_type]


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
# imports for generated wrapper code
import logging
{% if _this_node.profile %}import time{% endif %}
import math
from {{ plugin_name }} import ffi
import numpy as np
{% if _this_node.backend == 'GPU' %}import cupy as cp {% endif %}
from numpy.typing import NDArray
from gt4py.next.iterator.embedded import np_as_located_field
from icon4py.model.common.settings import xp

{% if _this_node.is_gt4py_program_present %}
# necessary imports when embedding a gt4py program directly
from gt4py.next import itir_python as run_roundtrip
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached, run_gtfn_gpu_cached
from icon4py.model.common.grid.simple import SimpleGrid

# We need a grid to pass offset providers to the embedded gt4py program (granules load their own grid at runtime)
grid = SimpleGrid()
{% endif %}

# logger setup
log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.{%- if _this_node.debug_mode -%}DEBUG{%- else -%}ERROR{%- endif -%},
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')
{% if _this_node.backend == 'GPU' %}logging.info(cp.show_config()) {% endif %}

# embedded module imports
{% for stmt in imports -%}
{{ stmt }}
{% endfor %}

# embedded function imports
{% for func in _this_node.functions -%}
from {{ module_name }} import {{ func.name }}
{% endfor %}

{% if _this_node.backend == 'GPU' %}
{{ cffi_unpack_gpu }}
{% else %}
{{ cffi_unpack }}
{% endif %}

{{ int_to_bool }}

{% for func in _this_node.functions %}

{{ cffi_decorator }}
def {{ func.name }}_wrapper(
{%- for arg in func.args -%}
{{ arg.name }}: {{ arg.py_type_hint | replace("KHalfDim","KDim") }}{% if not loop.last or func.global_size_args %}, {% endif %}
{%- endfor %}
{%- for arg in func.global_size_args -%}
{{ arg }}: int32{{ ", " if not loop.last else "" }}
{%- endfor -%}
):
    try:
        {%- if _this_node.debug_mode %}
        logging.info("Python Execution Context Start")
        {% endif %}

        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        unpack_start_time = time.perf_counter()
        {% endif %}

        # Unpack pointers into Ndarrays
        {% for arg in func.args %}
        {% if arg.is_array %}
        {%- if _this_node.debug_mode %}
        msg = '{{ arg.name }} before unpacking: %s' % str({{ arg.name}})
        logging.debug(msg)
        {% endif %}

        {%- if arg.name in _this_node.uninitialised_arrays -%}
        {{ arg.name }} = xp.ones((1,) * {{ arg.size_args_len }}, dtype={{arg.np_type}}, order="F")
        {%- else -%}
        {{ arg.name }} = unpack{%- if _this_node.backend == 'GPU' -%}_gpu{%- endif -%}({{ arg.name }}, {{ ", ".join(arg.size_args) }})
        {%- endif -%}

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

        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        unpack_end_time = time.perf_counter()
        logging.critical('{{ func.name }} unpacking and allocating arrays time per timestep: %s' % str(unpack_end_time - unpack_start_time))
        {% endif %}

        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        func_start_time = time.perf_counter()
        {% endif %}

        {{ func.name }}
        {%- if func.is_gt4py_program -%}.with_backend({{ _this_node.gt4py_backend }}){%- endif -%}(
        {%- for arg in func.args -%}
        {{ arg.name }}{{ ", " if not loop.last or func.is_gt4py_program else "" }}
        {%- endfor -%}
        {%- if func.is_gt4py_program -%}
        offset_provider=grid.offset_providers
        {%- endif -%}
        )

        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        func_end_time = time.perf_counter()
        logging.critical('{{ func.name }} function time per timestep: %s' % str(func_end_time - func_start_time))
        {% endif %}


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
        logging.critical("Python Execution Context End")
        {% endif %}

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0

        {% endfor %}


"""
    )


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""{{'\n'.join(functions)}}""")

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
    limited_area: bool
    dimension_positions: Sequence[DimensionPosition] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # call Func __post_init__
        self.dimension_positions = self.extract_dimension_positions()
        self.uninitialised_arrays = get_uninitialised_arrays(self.limited_area)

    def extract_dimension_positions(self) -> Sequence[DimensionPosition]:
        """Extract a unique set of dimension positions which are used to infer dimension sizes at runtime."""
        dim_positions: list[DimensionPosition] = []
        unique_size_args: set[str] = set()
        for arg in self.args:
            for index, size_arg in enumerate(arg.size_args):
                if size_arg not in unique_size_args:
                    dim_positions.append(
                        DimensionPosition(
                            variable=str(arg.name), size_arg=size_arg, index=index + 1
                        )
                    )  # Use Fortran indexing
                    unique_size_args.add(size_arg)
        return dim_positions


class F90Interface(Node):
    cffi_plugin: CffiPlugin
    limited_area: bool
    function_declaration: list[F90FunctionDeclaration] = datamodels.field(init=False)
    function_definition: list[F90FunctionDefinition] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        functions = self.cffi_plugin.functions
        self.function_declaration = [
            F90FunctionDeclaration(name=f.name, args=f.args, is_gt4py_program=f.is_gt4py_program)
            for f in functions
        ]
        self.function_definition = [
            F90FunctionDefinition(
                name=f.name,
                args=f.args,
                is_gt4py_program=f.is_gt4py_program,
                limited_area=self.limited_area,
            )
            for f in functions
        ]


class F90InterfaceGenerator(TemplatedGenerator):
    F90Interface = as_jinja(
        """\
module {{ _this_node.cffi_plugin.plugin_name }}
    use, intrinsic :: iso_c_binding
    implicit none

    {% for func in _this_node.cffi_plugin.functions %}
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

        return_code_param = ",&\nrc" if len(func.args) >= 1 else "rc"

        return self.generic_visit(
            func,
            assumed_size_array=False,
            param_names=arg_names,
            param_names_with_size_args=param_names_with_size_args,
            arrays=set([arg.name for arg in func.args if arg.is_array]).difference(
                set(func.uninitialised_arrays)
            ),
            return_code_param=return_code_param,
        )

    F90FunctionDefinition = as_jinja(
        """
subroutine {{name}}({{param_names}} {{ return_code_param }})
   use, intrinsic :: iso_c_binding
   {% for size_arg in global_size_args %}
   integer(c_int) :: {{ size_arg }}
   {% endfor %}
   {% for arg in args %}
   {{ arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code

   {% if arrays | length >= 1 %}
   !$ACC host_data use_device( &
   {%- for arr in arrays %}
       !$ACC {{ arr }}{% if not loop.last %}, &{% else %} &{% endif %}
   {%- endfor %}
   !$ACC )
   {% endif %}

   {% for d in _this_node.dimension_positions %}
   {{ d.size_arg }} = SIZE({{ d.variable }}, {{ d.index }})
   {% endfor %}

   rc = {{ name }}_wrapper({{ param_names_with_size_args }})

   {% if arrays | length >= 1 %}
   !$acc end host_data
   {% endif %}
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

    FuncParameter = as_jinja("""{{iso_c_type}}, {{dim}} {{value}} target :: {{name}}""")

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Sequence

import gt4py.next as gtx
from gt4py.eve import Node, datamodels
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from gt4py.next.type_system import type_specifications as ts

from icon4py.tools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
    BUILTIN_TO_NUMPY_TYPE,
)
from icon4py.tools.py2fgen.settings import GT4PyBackend
from icon4py.tools.py2fgen.utils import flatten_and_get_unique_elts


CFFI_DECORATOR = "@ffi.def_extern()"


class DimensionType(Node):
    name: str
    length: int


class FuncParameter(Node):
    name: str
    d_type: ts.ScalarKind
    dimensions: Sequence[gtx.Dimension]
    is_optional: bool = False
    size_args: list[str] = datamodels.field(init=False)
    is_array: bool = datamodels.field(init=False)
    is_bool: bool = datamodels.field(init=False)
    gtdims: list[str] = datamodels.field(init=False)
    size_args_len: int = datamodels.field(init=False)
    np_type: str = datamodels.field(init=False)
    domain: str = datamodels.field(init=False)

    def __post_init__(self) -> None:
        self.size_args = dims_to_size_strings(self.dimensions)
        self.size_args_len = len(self.size_args)
        self.is_array = True if len(self.dimensions) >= 1 else False
        self.is_bool = self.d_type == ts.ScalarKind.BOOL
        self.gtdims = [dim.value if not dim.value == "KHalf" else "K" for dim in self.dimensions]
        self.np_type = to_np_type(self.d_type)
        self.domain = (
            "{"
            + ",".join(f"{d}:{s}" for d, s in zip(self.gtdims, self.size_args, strict=True))
            + "}"
        )


class Func(Node):
    name: str
    args: Sequence[FuncParameter]
    global_size_args: Sequence[str] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.global_size_args = flatten_and_get_unique_elts(
            [dims_to_size_strings(arg.dimensions) for arg in self.args]
        )


class CffiPlugin(Node):
    module_name: str
    plugin_name: str
    functions: list[Func]


class PythonWrapper(CffiPlugin):
    backend: str
    debug_mode: bool
    profile: bool
    cffi_decorator: str = CFFI_DECORATOR
    gt4py_backend: str = datamodels.field(init=False)
    used_dimensions: set[gtx.Dimension] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.gt4py_backend = GT4PyBackend[self.backend].value

        self.used_dimensions = set()
        for func in self.functions:
            for arg in func.args:
                self.used_dimensions.update(arg.dimensions)


def to_c_type(scalar_type: ts.ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_np_type(scalar_type: ts.ScalarKind) -> str:
    """Convert a scalar type to its corresponding numpy type."""
    return BUILTIN_TO_NUMPY_TYPE[scalar_type]


def to_iso_c_type(scalar_type: ts.ScalarKind) -> str:
    """Convert a scalar type to its corresponding ISO C type."""
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: FuncParameter) -> Optional[str]:
    """
    Return the Fortran 90 'value' keyword for scalar types.

    Args:
        param: The function parameter to check.

    Returns:
        A string containing 'value,' for scalar types, otherwise an empty string.
    """
    return "value" if len(param.dimensions) == 0 else None


def render_c_pointer(param: FuncParameter) -> str:
    """Render a C pointer symbol for array types."""
    return "*" if len(param.dimensions) > 0 else ""


def render_fortran_array_dimensions(
    param: FuncParameter, assumed_size_array: bool
) -> Optional[str]:
    """
    Render Fortran array dimensions for array types.

    Args:
        param: The function parameter to check.

    Returns:
        A string representing Fortran array dimensions.
    """
    if len(param.dimensions) > 0 and assumed_size_array:
        return "dimension(*)"

    if len(param.dimensions) > 0 and not assumed_size_array:
        dims = ",".join(":" for _ in param.dimensions)
        return f"dimension({dims})"

    return None


def dims_to_size_strings(dimensions: Sequence[gtx.Dimension]) -> list[str]:
    """Convert Python array dimension values to Fortran array access strings.

    These already should be in Row-major order as defined in the Python function
    definition type hints for each array. These will be used to reshape the
    Column-major ordered arrays from Fortran in the unpack function.

    Args:
        dimensions: A sequence of dimensions to convert.

    Returns:
        A list of Fortran array size strings.
    """
    return [f"n_{dim.value}" for dim in dimensions]


def render_fortran_array_sizes(param: FuncParameter) -> Optional[str]:
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
    return None


class PythonWrapperGenerator(TemplatedGenerator):
    # TODO(havogt): put np_as_located_field logic into unpack
    PythonWrapper = as_jinja(
        """\
# imports for generated wrapper code
import logging
{% if _this_node.profile %}import time{% endif %}
from {{ plugin_name }} import ffi
{% if _this_node.backend == 'GPU' %}import cupy as cp {% endif %}
import gt4py.next as gtx
from gt4py.next.type_system import type_specifications as ts
from icon4py.tools.py2fgen.settings import config
from icon4py.tools.py2fgen import wrapper_utils
xp = config.array_ns

# logger setup
log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.{%- if _this_node.debug_mode -%}DEBUG{%- else -%}ERROR{%- endif -%},
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')
{% if _this_node.backend == 'GPU' %}logging.info(cp.show_config()) {% endif %}

# embedded function imports
{% for func in _this_node.functions -%}
from {{ module_name }} import {{ func.name }}
{% endfor %}

{% for dim in _this_node.used_dimensions -%}
{{ dim.value }} = gtx.Dimension("{{ dim.value }}", kind=gtx.DimensionKind.{{ dim.kind.upper() }})
{% endfor %}

{% for func in _this_node.functions %}

{{ cffi_decorator }}
def {{ func.name }}_wrapper(
{%- for arg in func.args -%}
{{ arg.name }}{% if not loop.last or func.global_size_args %}, {% endif %}
{%- endfor %}
{%- for arg in func.global_size_args -%}
{{ arg }}{{ ", " if not loop.last else "" }}
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


        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        allocate_start_time = time.perf_counter()
        {% endif %}

        {% for arg in func.global_size_args %}
        logging.info('size of {{ arg }} = %s' % str({{ arg }}))
        {% endfor %}

        # Convert ptr to GT4Py fields
        {% for arg in func.args %}
        {% if arg.is_array %}
        {{ arg.name }} = wrapper_utils.as_field(ffi, xp, {{ arg.name }}, ts.ScalarKind.{{ arg.d_type.name }}, {{arg.domain}}, {{arg.is_optional}})
        {% elif arg.is_bool %}
        assert isinstance({{ arg.name }}, int)
        {{ arg.name }} = {{ arg.name }} != 0
        {% endif %}
        {% endfor %}

        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        allocate_end_time = time.perf_counter()
        logging.critical('{{ func.name }} allocating to gt4py fields time per timestep: %s' % str(allocate_end_time - allocate_start_time))
        {% endif %}

        {% if _this_node.profile %}
        cp.cuda.Stream.null.synchronize()
        func_start_time = time.perf_counter()
        {% endif %}

        {{ func.name }}(
        {%- for arg in func.args -%}
        {{ arg.name }}{{ ", " if not loop.last else "" }}
        {%- endfor -%}
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
        msg = 'shape of {{ arg.name }} after computation = %s' % str({{ arg.name}}.shape if {{arg.name}} is not None else "None")
        logging.debug(msg)
        msg = '{{ arg.name }} after computation: %s' % str({{ arg.name }}.ndarray if {{ arg.name }} is not None else "None")
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

    def visit_FuncParameter(self, param: FuncParameter) -> str:
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
    dimension_positions: Sequence[DimensionPosition] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # call Func __post_init__
        self.dimension_positions = self.extract_dimension_positions()

    def extract_dimension_positions(self) -> Sequence[DimensionPosition]:
        """Extract a unique set of dimension positions which are used to infer dimension sizes at runtime."""
        dim_positions: list[DimensionPosition] = []
        unique_size_args: set[str] = set()
        non_optional_args = [arg for arg in self.args if not arg.is_optional]
        for arg in non_optional_args:
            for index, size_arg in enumerate(arg.size_args):
                if size_arg not in unique_size_args:
                    dim_positions.append(
                        DimensionPosition(
                            variable=str(arg.name), size_arg=size_arg, index=index + 1
                        )
                    )  # Use Fortran indexing
                    unique_size_args.add(size_arg)
        return dim_positions


def _render_parameter_declaration(name: str, attributes: Sequence[str | None]) -> str:
    return f"{','.join(attribute for attribute in  attributes if attribute is not None)} :: {name}"


class F90Interface(Node):
    cffi_plugin: CffiPlugin
    function_declaration: list[F90FunctionDeclaration] = datamodels.field(init=False)
    function_definition: list[F90FunctionDefinition] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        functions = self.cffi_plugin.functions
        self.function_declaration = [
            F90FunctionDeclaration(name=f.name, args=f.args) for f in functions
        ]
        self.function_definition = [
            F90FunctionDefinition(
                name=f.name,
                args=f.args,
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

    def visit_F90FunctionDeclaration(self, func: F90FunctionDeclaration, **kwargs: Any) -> str:
        param_names = ", &\n ".join(map(lambda x: x.name, func.args))
        if func.global_size_args:
            param_names += ",&\n" + ", &\n".join(func.global_size_args)

        param_declarations = [
            _render_parameter_declaration(
                name=param.name,
                attributes=[
                    "type(c_ptr)" if param.is_array else to_iso_c_type(param.d_type),
                    "value",
                    "target",
                ],
            )
            for param in func.args
        ]
        return self.generic_visit(
            func,
            as_func_declaration=True,
            param_names=param_names,
            param_declarations=param_declarations,
        )

    F90FunctionDeclaration = as_jinja(
        """
function {{name}}_wrapper({{param_names}}) bind(c, name="{{name}}_wrapper") result(rc)
   import :: c_int, c_double, c_bool, c_ptr
   {% for size_arg in global_size_args %}
   integer(c_int), value :: {{ size_arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code
   {% for param in param_declarations %}
   {{ param }}
   {% endfor %}
end function {{name}}_wrapper
    """
    )

    def visit_F90FunctionDefinition(self, func: F90FunctionDefinition, **kwargs: Any) -> str:
        def render_args(arg: FuncParameter) -> str:
            if arg.is_array and arg.is_optional:
                return f"{arg.name} = {arg.name}_ptr"
            elif arg.is_array:
                return f"{arg.name} = c_loc({arg.name})"
            return f"{arg.name} = {arg.name}"

        param_names = ", &\n ".join([arg.name for arg in func.args] + ["rc"])
        arg_names = [render_args(arg) for arg in func.args] + [
            f"{arg} = {arg}" for arg in func.global_size_args
        ]

        compiled_arg_names = ", &\n".join(arg_names)

        param_declarations = [
            _render_parameter_declaration(
                name=param.name,
                attributes=[
                    to_iso_c_type(param.d_type),
                    render_fortran_array_dimensions(param, False),
                    as_f90_value(param),
                    "pointer" if param.is_optional else "target",
                ],
            )
            for param in func.args
        ]

        return self.generic_visit(
            func,
            assumed_size_array=False,
            param_names=param_names,
            args_with_size_args=compiled_arg_names,
            non_optional_arrays=[
                arg.name for arg in func.args if arg.is_array if not arg.is_optional
            ],
            optional_arrays=[arg.name for arg in func.args if arg.is_array if arg.is_optional],
            as_allocatable=True,
            param_declarations=param_declarations,
            to_iso_c_type=to_iso_c_type,
            render_fortran_array_dimensions=render_fortran_array_dimensions,
        )

    F90FunctionDefinition = as_jinja(
        """
subroutine {{name}}({{param_names}})
   use, intrinsic :: iso_c_binding
   {% for size_arg in global_size_args %}
   integer(c_int) :: {{ size_arg }}
   {% endfor %}
   {% for arg in param_declarations %}
   {{ arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code
   ! ptrs
   {% for arg in _this_node.args if arg.is_optional %}
   type(c_ptr) :: {{ arg.name }}_ptr
   {% endfor %}

   {% for arg in _this_node.args if arg.is_optional %}
   {{ arg.name }}_ptr = c_null_ptr
   {% endfor %}

   {%- for arr in non_optional_arrays %}
       !$acc host_data use_device({{ arr }})
   {%- endfor %}
   {%- for arr in optional_arrays %}
       !$acc host_data use_device({{ arr }}) if(associated({{ arr }}))
   {%- endfor %}

   {% for d in _this_node.dimension_positions %}
   {{ d.size_arg }} = SIZE({{ d.variable }}, {{ d.index }})
   {% endfor %}
   
   {% for arg in _this_node.args if arg.is_optional %}
   if(associated({{ arg.name }})) then
   {{ arg.name }}_ptr = c_loc({{ arg.name }})
    endif
   {% endfor %}

   rc = {{ name }}_wrapper({{ args_with_size_args }})

   {%- for arr in non_optional_arrays %}
   !$acc end host_data
   {%- endfor %}
   {%- for arr in optional_arrays %}
   !$acc end host_data
   {%- endfor %}
end subroutine {{name}}
    """
    )

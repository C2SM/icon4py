# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Sequence, TypeGuard, Union

from gt4py.eve import Node, datamodels
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from gt4py.next.type_system import type_specifications as ts

from icon4py.tools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
    BUILTIN_TO_NUMPY_TYPE,
)
from icon4py.tools.py2fgen import _definitions


CFFI_DECORATOR = "@ffi.def_extern()"


def is_array(param: _definitions.ParamDescriptor) -> TypeGuard[_definitions.ArrayParamDescriptor]:
    return isinstance(param, _definitions.ArrayParamDescriptor)


class Func(Node):
    name: str
    args: dict[str, Union[_definitions.ArrayParamDescriptor, _definitions.ScalarParamDescriptor]]
    rendered_params: str = datamodels.field(init=False)  # TODO remove

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        params = []
        for name, param in self.args.items():
            params.append(name)
            if is_array(param):
                for i in range(param.rank):
                    params.append(_size_arg_name(name, i))
        params.append("on_gpu")
        self.rendered_params = ", ".join(params)


class CffiPlugin(Node):
    module_name: str
    plugin_name: str
    functions: list[Func]


class PythonWrapper(CffiPlugin):
    cffi_decorator: str = CFFI_DECORATOR


def to_c_type(scalar_type: ts.ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_np_type(scalar_type: ts.ScalarKind) -> str:
    """Convert a scalar type to its corresponding numpy type."""
    return BUILTIN_TO_NUMPY_TYPE[scalar_type]


def to_iso_c_type(scalar_type: ts.ScalarKind) -> str:
    """Convert a scalar type to its corresponding ISO C type."""
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: _definitions.ParamDescriptor) -> Optional[str]:
    """
    Return the Fortran 90 'value' keyword for scalar types.

    Args:
        param: The function parameter to check.

    Returns:
        A string containing 'value,' for scalar types, otherwise an empty string.
    """
    return "value" if not is_array(param) else None


def render_c_pointer(param: _definitions.ParamDescriptor) -> str:
    """Render a C pointer symbol for array types."""
    return "*" if is_array(param) else ""


def render_fortran_array_dimensions(
    param: _definitions.ParamDescriptor, assumed_size_array: bool
) -> Optional[str]:
    """
    Render Fortran array dimensions for array types.

    Args:
        param: The function parameter to check.

    Returns:
        A string representing Fortran array dimensions.
    """
    if is_array(param):
        if assumed_size_array:
            return "dimension(*)"
        else:
            dims = ",".join(":" for _ in range(param.rank))
        return f"dimension({dims})"

    return None


class PythonWrapperGenerator(TemplatedGenerator):
    def visit_PythonWrapper(self, node: PythonWrapper, **kwargs: Any) -> str:
        def render_size_args_tuple(name: str, param: _definitions.ArrayParamDescriptor) -> str:
            size_args = ",".join(f"{_size_arg_name(name, i)}" for i in range(param.rank))
            return f"({size_args},)"

        # TODO remove ts
        return self.generic_visit(
            node, ts=ts, is_array=is_array, render_size_args_tuple=render_size_args_tuple, **kwargs
        )

    PythonWrapper = as_jinja(
        """\
import logging
from gt4py.next.type_system.type_specifications import ScalarKind
from {{ plugin_name }} import ffi
from icon4py.tools.py2fgen import wrapper_utils, runtime_config, _runtime

if __debug__:
    logger = logging.getLogger(__name__)
    log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, runtime_config.LOG_LEVEL),
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')


# embedded function imports
{% for func in _this_node.functions -%}
from {{ module_name }} import {{ func.name }}
{% endfor %}

{% for func in _this_node.functions %}

{{ cffi_decorator }}
def {{ func.name }}_wrapper(
{{func.rendered_params}}
):
    try:
        if __debug__:
            logger.info("Python execution of {{ func.name }} started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # Convert ptrs 
        {% for name, arg in func.args.items() %}
        {% if is_array(arg) %}
        # TODO use the size_args function
        {{ name }} = ({{ name }}, ({{ render_size_args_tuple(name, arg) }},), {% if arg.device == "host" %}False{% else %}on_gpu{% endif %}, {{ arg.is_optional }})
        {% elif arg.dtype == ts.ScalarKind.BOOL %}
        # TODO move bool translation to postprocessing
        assert isinstance({{ name }}, int)
        {{ name }} = {{ name }} != 0
        {% endif %}
        {% endfor %}

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info('{{ func.name }} constructing `ArrayDescriptors` time: %s' % str(allocate_end_time - unpack_start_time))

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            meta = {}
        else:
            meta = None
        {{ func.name }}(
        ffi = ffi,
        meta = meta,
        {%- for name, arg in func.args.items() -%}
        {{ name }} = {{ name }}{{ "," }}
        {%- endfor -%}
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info('{{ func.name }} convert time: %s' % str(meta["convert_end_time"] - meta["convert_start_time"]))
                logger.info('{{ func.name }} execution time: %s' % str(func_end_time - func_start_time))


        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):
                {% for name, arg in func.args.items() %}
                {% if is_array(arg) %}
                msg = 'shape of {{ name }} after computation = %s' % str({{ name}}.shape if {{name}} is not None else "None")
                logger.debug(msg)
                msg = '{{ name }} after computation: %s' % str(wrapper_utils.as_array(ffi, {{ name }}, {{ arg.dtype }}) if {{ name }} is not None else "None")
                logger.debug(msg)
                {% endif %}
                {% endfor %}

        if __debug__:
            logger.info("Python execution of {{ func.name }} completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0

        {% endfor %}


"""
    )


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""{{'\n'.join(functions)}}""")

    def visit_Func(self, func: Func) -> str:
        params = []
        for name, param in func.args.items():
            params.append(self.visit_Parameter(name, param))
            if is_array(param):
                for i in range(param.rank):
                    params.append(f"int {_size_arg_name(name, i)}")
        params.append("int on_gpu")
        rendered_params = ", ".join(params)
        return self.generic_visit(func, rendered_params=rendered_params)

    Func = as_jinja("extern int {{ name }}_wrapper({{rendered_params}});")

    def visit_Parameter(self, name: str, param: _definitions.ParamDescriptor, **kwargs: Any) -> str:
        return f"{to_c_type(param.dtype)}{render_c_pointer(param)} {name}"


class F90FunctionDeclaration(Func):
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # call Func __post_init__


class F90FunctionDefinition(Func):
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # call Func __post_init__


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


def _size_arg_name(name: str, i: int) -> str:
    return f"{name}_size_{i}"


def _size_param_declaration(name: str, value: bool = True) -> str:
    return f"integer(c_int){', value' if value else ''} :: {name}"


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
        param_names = []
        param_declarations = []
        for name, param in func.args.items():
            param_names.append(name)
            param_declarations.append(
                _render_parameter_declaration(
                    name=name,
                    attributes=[
                        "type(c_ptr)" if is_array(param) else to_iso_c_type(param.dtype),
                        "value",
                        "target",
                    ],
                )
            )
            if is_array(param):
                for i in range(param.rank):
                    size_name = _size_arg_name(name, i)
                    param_names.append(size_name)
                    param_declarations.append(_size_param_declaration(size_name))

        # on_gpu flag
        param_declarations.append("logical(c_int), value :: on_gpu")
        param_names.append("on_gpu")

        param_names_str = ", &\n ".join(param_names)

        return self.generic_visit(
            func,
            as_func_declaration=True,
            param_names=param_names_str,
            param_declarations=param_declarations,
        )

    F90FunctionDeclaration = as_jinja(
        """
function {{name}}_wrapper({{param_names}}) bind(c, name="{{name}}_wrapper") result(rc)
   import :: c_int, c_double, c_bool, c_ptr
   integer(c_int) :: rc  ! Stores the return code
   {% for param in param_declarations %}
   {{ param }}
   {% endfor %}
end function {{name}}_wrapper
    """
    )

    def visit_F90FunctionDefinition(self, func: F90FunctionDefinition, **kwargs: Any) -> str:
        def render_args(name: str, param: _definitions.ParamDescriptor) -> str:
            if is_array(param):
                if param.is_optional:
                    return f"{name} = {name}_ptr"
                else:
                    return f"{name} = c_loc({name})"
            return f"{name} = {name}"

        param_names = ", &\n ".join([name for name in func.args.keys()] + ["rc"])
        args = []
        for name, param in func.args.items():
            args.append(render_args(name, param))
            if is_array(param):
                for i in range(param.rank):
                    size_name = _size_arg_name(name, i)
                    args.append(f"{size_name} = {size_name}")

        # on_gpu flag
        args.append("on_gpu = on_gpu")
        compiled_arg_names = ", &\n".join(args)

        param_declarations = [
            _render_parameter_declaration(
                name=name,
                attributes=[
                    to_iso_c_type(param.dtype),
                    render_fortran_array_dimensions(param, False),
                    as_f90_value(param),
                    "pointer" if is_array(param) and param.is_optional else "target",
                ],
            )
            for name, param in func.args.items()
        ]

        # on_gpu flag
        param_declarations.append("logical(c_int) :: on_gpu")

        def get_sizes_maker(name: str, param: _definitions.ArrayParamDescriptor) -> str:
            return "\n".join(
                f"{_size_arg_name(name, i)} = SIZE({name}, {i + 1})" for i in range(param.rank)
            )

        for name, param in func.args.items():
            if is_array(param):
                for i in range(param.rank):
                    size_name = _size_arg_name(name, i)
                    param_declarations.append(_size_param_declaration(size_name, value=False))

        return self.generic_visit(
            func,
            assumed_size_array=False,
            param_names=param_names,
            args_with_size_args=compiled_arg_names,
            non_optional_arrays=[
                name
                for name, param in func.args.items()
                if is_array(param)
                and not param.is_optional
                and param.device == _definitions.DeviceType.MAYBE_DEVICE
            ],
            optional_arrays=[
                name
                for name, param in func.args.items()
                if is_array(param)
                and param.is_optional
                and param.device == _definitions.DeviceType.MAYBE_DEVICE
            ],
            as_allocatable=True,
            param_declarations=param_declarations,
            to_iso_c_type=to_iso_c_type,
            render_fortran_array_dimensions=render_fortran_array_dimensions,
            get_sizes_maker=get_sizes_maker,
            is_array=is_array,
        )

    F90FunctionDefinition = as_jinja(
        """
subroutine {{name}}({{param_names}})
   use, intrinsic :: iso_c_binding
   {% for arg in param_declarations %}
   {{ arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code
   ! ptrs
   {% for name in optional_arrays %}
   type(c_ptr) :: {{ name }}_ptr
   {% endfor %}

   {% for name in optional_arrays %}
   {{ name }}_ptr = c_null_ptr
   {% endfor %}

   {%- for arr in non_optional_arrays %}
       !$acc host_data use_device({{ arr }})
   {%- endfor %}
   {%- for arr in optional_arrays %}
       !$acc host_data use_device({{ arr }}) if(associated({{ arr }}))
   {%- endfor %}
   
   #ifdef _OPENACC
   on_gpu = .True.
   #else
   on_gpu = .False.
   #endif

   {% for name, param in _this_node.args.items() if is_array(param) and not param.is_optional %}
     {{ get_sizes_maker(name, param) }}
   {% endfor %}

   {% for name, param in _this_node.args.items() if is_array(param) and param.is_optional %}
   if(associated({{ name }})) then
   {{ name }}_ptr = c_loc({{ name }})
     {{ get_sizes_maker(name, param) }}
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

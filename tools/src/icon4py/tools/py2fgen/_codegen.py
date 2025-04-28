# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Final, Literal, Sequence, TypeGuard, Union

from gt4py.eve import Node, codegen
from gt4py.eve.codegen import JinjaTemplate as as_jinja

from icon4py.tools.py2fgen import _definitions, _utils


CFFI_DECORATOR = "@ffi.def_extern()"

BUILTIN_TO_ISO_C_TYPE: Final[dict[_definitions.ScalarKind, str]] = {
    _definitions.FLOAT64: "real(c_double)",
    _definitions.FLOAT32: "real(c_float)",
    _definitions.BOOL: "logical(c_int)",
    _definitions.INT32: "integer(c_int)",
    _definitions.INT64: "integer(c_long)",
}
BUILTIN_TO_CPP_TYPE: Final[dict[_definitions.ScalarKind, str]] = {
    _definitions.FLOAT64: "double",
    _definitions.FLOAT32: "float",
    _definitions.BOOL: "int",
    _definitions.INT32: "int",
    _definitions.INT64: "long",
}
BUILTIN_TO_NUMPY_TYPE: Final[dict[_definitions.ScalarKind, str]] = {
    _definitions.FLOAT64: "xp.float64",
    _definitions.FLOAT32: "xp.float32",
    _definitions.BOOL: "xp.int32",
    _definitions.INT32: "xp.int32",
    _definitions.INT64: "xp.int64",
}


def is_array(param: _definitions.ParamDescriptor) -> TypeGuard[_definitions.ArrayParamDescriptor]:
    return isinstance(param, _definitions.ArrayParamDescriptor)


class Func(Node):
    name: str
    args: dict[str, Union[_definitions.ArrayParamDescriptor, _definitions.ScalarParamDescriptor]]


class BindingsLibrary(Node):
    module_name: str
    library_name: str
    functions: list[Func]


def to_c_type(scalar_type: _definitions.ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_np_type(scalar_type: _definitions.ScalarKind) -> str:
    """Convert a scalar type to its corresponding numpy type."""
    return BUILTIN_TO_NUMPY_TYPE[scalar_type]


def to_iso_c_type(scalar_type: _definitions.ScalarKind) -> str:
    """Convert a scalar type to its corresponding ISO C type."""
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: _definitions.ParamDescriptor) -> Literal["value", None]:
    """
    Return the Fortran 90 'value' keyword for scalar types.

    Args:
        param: The function parameter to check.

    Returns:
        A string containing 'value' for scalar types, otherwise an empty string.
    """
    return "value" if not is_array(param) else None


def render_c_pointer(param: _definitions.ParamDescriptor) -> Literal["*", ""]:
    """Render a C pointer symbol for array types."""
    return "*" if is_array(param) else ""


def render_fortran_array_dimensions(
    param: _definitions.ParamDescriptor, assumed_size_array: bool
) -> str | None:
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


class PythonWrapperGenerator(codegen.TemplatedGenerator):
    def visit_BindingsLibrary(self, node: BindingsLibrary, **kwargs: Any) -> str:
        def render_size_args_tuple(name: str, param: _definitions.ArrayParamDescriptor) -> str:
            size_args = ",".join(f"{_size_arg_name(name, i)}" for i in range(param.rank))
            return f"({size_args},)"

        def render_params(func: Func) -> str:
            params = []
            for name, param in func.args.items():
                params.append(name)
                if is_array(param):
                    params.extend(_size_arg_name(name, i) for i in range(param.rank))
            params.append("on_gpu")
            return ", ".join(params)

        return self.generic_visit(
            node,
            ScalarKind=_definitions.ScalarKind,
            is_array=is_array,
            render_size_args_tuple=render_size_args_tuple,
            render_params=render_params,
            cffi_decorator=CFFI_DECORATOR,
            **kwargs,
        )

    BindingsLibrary = as_jinja(
        """\
import logging
from {{ library_name }} import ffi
from icon4py.tools.py2fgen import runtime_config, _runtime, _definitions, _conversion

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
{{render_params(func)}}
):
    try:
        if __debug__:
            logger.info("Python execution of {{ func.name }} started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayInfos
        {% for name, arg in func.args.items() %}
        {% if is_array(arg) %}
        {{ name }} = ({{ name }}, {{ render_size_args_tuple(name, arg) }}, {% if arg.memory_space == "host" %}False{% else %}on_gpu{% endif %}, {{ arg.is_optional }})
        {% endif %}
        {% endfor %}

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info('{{ func.name }} constructing `ArrayInfos` time: %s' % str(allocate_end_time - unpack_start_time))

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            perf_counters = {}
        else:
            perf_counters = None
        {{ func.name }}(
        ffi = ffi,
        perf_counters = perf_counters,
        {%- for name, arg in func.args.items() -%}
        {{ name }} = {{ name }}{{ "," }}
        {%- endfor -%}
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info('{{ func.name }} convert time: %s' % str(perf_counters["convert_end_time"] - perf_counters["convert_start_time"]))
                logger.info('{{ func.name }} execution time: %s' % str(func_end_time - func_start_time))


        {% if func.args %}
        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):
                {% for name, arg in func.args.items() %}
                {% if is_array(arg) %}
                msg = 'shape of {{ name }} after computation = %s' % str({{ name}}.shape if {{name}} is not None else "None")
                logger.debug(msg)
                msg = '{{ name }} after computation: %s' % str(_conversion.as_array(ffi, {{ name }}, _definitions.{{ arg.dtype.name }}) if {{ name }} is not None else "None")
                logger.debug(msg)
                {% endif %}
                {% endfor %}
        {% endif %}

        if __debug__:
            logger.info("Python execution of {{ func.name }} completed.")

    except Exception as e:
        if __debug__:
            logger.exception(f"A Python error occurred: {e}")
        else:
            print(f"A Python error occurred: {e}", flush=True)
        return 1

    return 0

        {% endfor %}


"""
    )


class CHeaderGenerator(codegen.TemplatedGenerator):
    BindingsLibrary = as_jinja("{{'\n'.join(functions)}}")

    def visit_Func(self, func: Func) -> str:
        params = []
        for name, param in func.args.items():
            params.append(self.visit_Parameter(name, param))
            if is_array(param):
                params.extend(f"int {_size_arg_name(name, i)}" for i in range(param.rank))
        params.append("int on_gpu")

        rendered_params = ", ".join(params)
        return self.generic_visit(func, rendered_params=rendered_params)

    Func = as_jinja("extern int {{ name }}_wrapper({{rendered_params}});")

    def visit_Parameter(self, name: str, param: _definitions.ParamDescriptor, **kwargs: Any) -> str:
        return f"{to_c_type(param.dtype)}{render_c_pointer(param)} {name}"


def _render_parameter_declaration(name: str, attributes: Sequence[str | None]) -> str:
    return f"{','.join(attribute for attribute in  attributes if attribute is not None)} :: {name}"


def _size_arg_name(name: str, i: int) -> str:
    return f"{name}_size_{i}"


def _size_param_declaration(name: str, value: bool = True) -> str:
    return f"integer(c_int){', value' if value else ''} :: {name}"


class FortranISOCBindingsGenerator(codegen.TemplatedGenerator):
    def visit_Func(self, func: Func, **kwargs: Any) -> str:
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

    Func = as_jinja(
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


class FortranBindingsFunctionGenerator(codegen.TemplatedGenerator):
    def visit_Func(self, func: Func, **kwargs: Any) -> str:
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
                args.extend(
                    f"{_size_arg_name(name, i)} = {_size_arg_name(name, i)}"
                    for i in range(param.rank)
                )

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
                and param.memory_space == _definitions.MemorySpace.MAYBE_DEVICE
            ],
            optional_arrays=[
                name
                for name, param in func.args.items()
                if is_array(param)
                and param.is_optional
                and param.memory_space == _definitions.MemorySpace.MAYBE_DEVICE
            ],
            as_allocatable=True,
            param_declarations=param_declarations,
            to_iso_c_type=to_iso_c_type,
            render_fortran_array_dimensions=render_fortran_array_dimensions,
            get_sizes_maker=get_sizes_maker,
            is_array=is_array,
        )

    Func = as_jinja(
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


class F90InterfaceGenerator(codegen.TemplatedGenerator):
    def visit_BindingsLibrary(self, bindings_library: BindingsLibrary, **kwargs: Any) -> str:
        bindings_functions = FortranBindingsFunctionGenerator.apply(bindings_library.functions)
        iso_c_bindings_functions = FortranISOCBindingsGenerator.apply(bindings_library.functions)
        return self.generic_visit(
            bindings_library,
            bindings_functions=bindings_functions,
            iso_c_bindings_functions=iso_c_bindings_functions,
        )

    BindingsLibrary = as_jinja(
        """\
module {{ _this_node.library_name }}
    use, intrinsic :: iso_c_binding
    implicit none

    {% for func in _this_node.functions %}
    public :: {{ func.name }}
    {% endfor %}

interface
    {{ '\n'.join(iso_c_bindings_functions) }}
end interface

contains
    {{ '\n'.join(bindings_functions) }}
end module
"""
    )


def generate_c_header(bindings_library: BindingsLibrary) -> str:
    """
    Generate C header code from the given plugin.

    Args:
        plugin: The BindingsLibrary instance containing information for code generation.

    Returns:
        Formatted C header code as a string.
    """
    generated_code = CHeaderGenerator.apply(bindings_library)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def generate_python_wrapper(bindings_library: BindingsLibrary) -> str:
    """
    Generate Python wrapper code.

    Args:
        plugin: The BindingsLibrary instance containing information for code generation.

    Returns:
        Formatted Python wrapper code as a string.
    """
    generated_code = PythonWrapperGenerator.apply(bindings_library)
    return codegen.format_source("python", generated_code)


def generate_f90_interface(bindings_library: BindingsLibrary) -> str:
    """
    Generate Fortran 90 interface code.

    Args:
        plugin: The BindingsLibrary instance containing information for code generation.
    """
    generated_code = F90InterfaceGenerator.apply(bindings_library)
    return _utils.format_fortran_code(generated_code)

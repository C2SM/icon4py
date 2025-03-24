# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import types
import typing
from collections.abc import Mapping
from typing import Any, Callable, Optional

import cffi

from icon4py.tools.py2fgen import _conversion, _definitions, _runtime


# TODO(egparedes): possibly use `TypeForm` for the annotation parameter,
# once https://peps.python.org/pep-0747/ is approved.
def _from_annotated(annotation: Any) -> _definitions.ParamDescriptor | None:
    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(meta, _definitions.ParamDescriptor):
                return meta
    return None


def param_descriptor_from_annotation(
    annotation: Any,
    annotation_descriptor_hook: Optional[Callable[[Any], _definitions.ParamDescriptor | None]],
) -> _definitions.ParamDescriptor:
    descriptor = None
    if annotation_descriptor_hook is not None:
        descriptor = annotation_descriptor_hook(annotation)
    if descriptor is None:
        descriptor = _from_annotated(annotation)
    if descriptor is None:
        raise ValueError(f"Could not determine descriptor for type annotation {annotation}.")
    return descriptor


def get_param_descriptors(
    type_hints: dict[str, Any],
    param_descriptors: Optional[_definitions.ParamDescriptors],
    annotation_descriptor_hook: Optional[Callable[[Any], _definitions.ParamDescriptor]],
) -> _definitions.ParamDescriptors:
    if param_descriptors is not None:
        if annotation_descriptor_hook is not None:
            raise ValueError(
                "Cannot pass both 'param_descriptors' and 'annotation_descriptor_hook'."
            )
        return param_descriptors

    return {
        name: param_descriptor_from_annotation(annotation, annotation_descriptor_hook)
        for name, annotation in type_hints.items()
    }


def get_param_mappings(
    type_hints: dict[str, Any],
    annotation_mapping_hook: Callable[
        [Any, _definitions.ParamDescriptor], _definitions.MapperType | None
    ]
    | None,
    param_descriptors: _definitions.ParamDescriptors,
) -> Mapping[str, _definitions.MapperType]:
    mappings: dict[str, _definitions.MapperType] = {}
    for name, annotation in type_hints.items():
        if annotation_mapping_hook is not None:
            mapping = annotation_mapping_hook(annotation, param_descriptors[name])
            if mapping is None:
                mapping = _conversion.default_mapping(annotation, param_descriptors[name])
            if mapping is not None:
                mappings[name] = mapping
    return mappings


# Note that the mapping functions are performance relevant and should be cached (not the hook itself)
@dataclasses.dataclass
class _DecoratedFunction:
    """
    Wraps a function to make it exportable with 'py2fgen'.

    A function is exportable if it provides the attribute 'param_descriptors'.

    See :func:`export` for details.
    """

    _fun: Callable
    annotation_descriptor_hook: Optional[Callable]  # TODO type annotation
    annotation_mapping_hook: Optional[Callable]  # TODO type annotation
    param_descriptors: Optional[_definitions.ParamDescriptors]
    _mapping: Mapping[str, Callable] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        type_hints = typing.get_type_hints(self._fun, include_extras=True)
        if "return" in type_hints:
            if type_hints["return"] is types.NoneType:
                type_hints.pop("return")
            else:
                raise ValueError(
                    "Exported functions must not have a return type. Use 'None' instead."
                )

        self.param_descriptors = get_param_descriptors(
            type_hints, self.param_descriptors, self.annotation_descriptor_hook
        )

        self._mapping = get_param_mappings(
            type_hints, self.annotation_mapping_hook, self.param_descriptors
        )

    def __call__(self, ffi: cffi.FFI, meta: Optional[dict], **kwargs: Any) -> Any:
        # Notes: For performance reasons we could switch to positional-only arguments
        # (this is purely internal between the generated Python code and this function).
        # However, an experiment showed that this had small impact.
        # Re-evaluate if we need to tweak performance.
        if __debug__ and meta is not None:
            meta["convert_start_time"] = _runtime.perf_counter()
        kwargs = {
            k: self._mapping[k](v, ffi=ffi) if k in self._mapping else v for k, v in kwargs.items()
        }
        if __debug__ and meta is not None:
            meta["convert_end_time"] = _runtime.perf_counter()
        return self._fun(**kwargs)


def export(
    annotation_descriptor_hook: Optional[Callable] = None,
    annotation_mapping_hook: Optional[Callable] = None,
    param_descriptors: Optional[_definitions.ParamDescriptors] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator to mark a function as exportable.

    The standard mechanism for exporting a function is to decorate the function with
    '@py2fgen.export(param_descriptors=...)'. Where 'ParamDescriptors' is a dictionary
    that provides a :class:`ParamDescriptor` for each parameter of the function.

    Additionally, the user can provide a hock to fill 'param_descriptors' from the parameters
    type annotations.

    For runtime processing of a scalar parameter or an 'ArrayInfo', the user can provide a hook
    which provides a mapping function to translate the parameter to a different type, e.g.
    to translate 'ArrayInfo' to a NumPy array.
    Note: The mapping function (not the hook) is called at every invocation of the function,
    therefore it is recommended to use a cache for the mapping function.

    A default mapping is provided, see :func:`_conversion.default_mapping`.
    """

    # precise typing is difficult (impossible?) since we are manipulating the args
    def impl(fun: Callable) -> Callable:
        return functools.update_wrapper(
            _DecoratedFunction(
                fun,
                annotation_descriptor_hook=annotation_descriptor_hook,
                annotation_mapping_hook=annotation_mapping_hook,
                param_descriptors=param_descriptors,
            ),
            fun,
        )

    return impl

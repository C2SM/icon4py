# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Optional, TypeAlias

import cffi
from gt4py.next.type_system import type_specifications as gtx_ts  # TODO use py2fgen types

from icon4py.tools.py2fgen import _definitions, _runtime, wrapper_utils


def _from_annotated(annotation: Any) -> Optional[_definitions.ParamDescriptor]:
    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(
                meta, (_definitions.ArrayParamDescriptor, _definitions.ScalarParamDescriptor)
            ):
                return meta
    return None


def param_descriptor_from_annotation(
    annotation: Any,
    annotation_descriptor_hook: Optional[Callable[[Any], _definitions.ParamDescriptor]],
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
    signature: inspect.Signature,
    param_descriptors: Optional[_definitions.ParamDescriptors],
    annotation_descriptor_hook: Optional[Callable[[Any], _definitions.ParamDescriptor]],
) -> _definitions.ParamDescriptors:
    if param_descriptors is not None:
        if annotation_descriptor_hook is not None:
            raise ValueError(
                "Cannot pass both 'param_descriptors' and 'annotation_descriptor_hook'."
            )
        return param_descriptors

    params = {}
    for name, param in signature.parameters.items():
        params[name] = param_descriptor_from_annotation(
            param.annotation, annotation_descriptor_hook
        )

    return params


NDArray = Any  # = np.ndarray | cp.ndarray (conditionally) TODO move to better place and fix


def _as_array(
    dtype: gtx_ts.ScalarKind,
) -> Callable[[wrapper_utils.ArrayDescriptor, cffi.FFI], NDArray]:
    @functools.lru_cache(maxsize=None)
    def impl(array_descriptor: wrapper_utils.ArrayDescriptor, *, ffi: cffi.FFI) -> NDArray:
        return wrapper_utils.as_array(ffi, array_descriptor, dtype)

    return impl


MapperType: TypeAlias = Callable[[wrapper_utils.ArrayDescriptor, cffi.FFI], Any]


def default_mapping(_: Any, param_descriptor: _definitions.ParamDescriptor) -> MapperType | None:
    """Translates ArrayDescriptor into (Numpy or Cupy) NDArray."""
    if isinstance(param_descriptor, _definitions.ArrayParamDescriptor):
        return _as_array(param_descriptor.dtype)
    return None


def get_param_mappings(
    signature: inspect.Signature,
    annotation_mapping_hook: Callable[[Any, _definitions.ParamDescriptor], MapperType] | None,
    param_descriptors: _definitions.ParamDescriptors,
) -> Mapping[str, MapperType]:
    mappings: dict[str, MapperType] = {}
    for name, param in signature.parameters.items():
        if annotation_mapping_hook is not None:
            mapping = annotation_mapping_hook(param.annotation, param_descriptors[name])
            if mapping is None:
                mapping = default_mapping(param.annotation, param_descriptors[name])
            if mapping is not None:
                mappings[name] = mapping
    return mappings


# TODO move to docstring
# To generate bindings either
# - pass a full function descriptor
# - pass an annotation_descriptor_hook which receives the annotation and return the FuncParameter/ArrayParameter
# To postprocess the arguments coming from Fortran
# - pass a annotation_mapping_hook which receives the annotation and returns a function that takes the ArrayDescriptor (TODO extend to any descriptor) and ffi
# Note that the mapping functions are performance relevant and should be cached (not the hook itself)
@dataclasses.dataclass
class _DecoratedFunction:
    _fun: Callable
    annotation_descriptor_hook: Optional[Callable]  # TODO type annotation
    annotation_mapping_hook: Optional[Callable]  # TODO type annotation
    param_descriptors: Optional[_definitions.ParamDescriptors]
    _mapping: Mapping[str, Callable] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        signature = inspect.signature(self._fun)
        self.param_descriptors = get_param_descriptors(
            signature, self.param_descriptors, self.annotation_descriptor_hook
        )

        self._mapping = get_param_mappings(
            signature, self.annotation_mapping_hook, self.param_descriptors
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

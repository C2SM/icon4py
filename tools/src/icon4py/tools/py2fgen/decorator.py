# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO module needs better name!!!

import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence

from gt4py import next as gtx
from gt4py.next.type_system import type_specifications as gtx_ts

from icon4py.tools.py2fgen import parsing, template, wrapper_utils


def _as_field(dims: Sequence[gtx.Dimension], scalar_kind: gtx_ts.ScalarKind) -> Callable:
    @functools.lru_cache(maxsize=None)
    def impl(array_descriptor: wrapper_utils.ArrayDescriptor) -> Optional[gtx.Field]:
        domain = {d: s for d, s in zip(dims, array_descriptor.shape, strict=True)}
        return wrapper_utils.as_field(
            array_descriptor.ffi,
            array_descriptor.on_gpu,
            array_descriptor.ptr,
            scalar_kind,
            domain,
            is_optional=False,
        )

    return impl


@dataclasses.dataclass
class _DecoratedFunction:
    _fun: Callable
    function_descriptor: template.Func = dataclasses.field(init=False)
    _mapping: dict[str, Callable] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        params = parsing._parse_params(self._fun)
        self.function_descriptor = template.Func(name=self._fun.__name__, args=params)

        mapping = {}
        for param in self.function_descriptor.args:
            if param.is_array:
                mapping[param.name] = _as_field(param.dimensions, param.d_type)
        self._mapping = mapping

    def __call__(self, **kwargs: Any) -> Any:  # TODO switch to positional arguments for performance
        kwargs = {
            k: self._mapping[k](v) if k in self._mapping else v for k, v in kwargs.items()
        }  # TODO possibly cache this
        return self._fun(**kwargs)


def export(fun: Callable) -> Callable:
    return functools.update_wrapper(_DecoratedFunction(fun), fun)

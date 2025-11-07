# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import gt4py.next as gtx
import gt4py.next.allocators as gtx_allocators
import gt4py.next.typing as gtx_typing


try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None


def is_cupy_device(allocator: gtx_typing.FieldBufferAllocationUtil | None) -> bool:
    if allocator is None:
        return False

    if gtx.CUPY_DEVICE_TYPE is None:
        return False

    return gtx_allocators.is_field_allocation_tool_for(allocator, gtx.CUPY_DEVICE_TYPE)  # type: ignore [type-var] #gt4py-related typing


def sync(allocator: gtx_typing.FieldBufferAllocationUtil | None = None) -> None:
    """
    Synchronize the device if appropriate for the given backend.

    Note: this is and ad-hoc interface, maybe the function should get the device to sync for.
    """
    if allocator is not None and is_cupy_device(allocator):
        cp.cuda.runtime.deviceSynchronize()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def synchronized_function(
    func: Callable[_P, _R], *, allocator: gtx_typing.FieldBufferAllocationUtil | None
) -> Callable[_P, _R]:
    """
    Wraps a function and synchronizes after execution
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _R:
        result = func(*args, **kwargs)
        sync(allocator=allocator)
        return result

    return wrapper


def synchronized(
    allocator: gtx_typing.FieldBufferAllocationUtil | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that synchronizes the device after the function execution.
    """
    return functools.partial(synchronized_function, allocator=allocator)

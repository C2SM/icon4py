# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from collections.abc import Callable
from typing import Any, TypeGuard

import gt4py.next as gtx
import gt4py.next.allocators as gtx_allocators
import gt4py.next.typing as gtx_typing

from icon4py.model.common import model_backends


try:
    import cupy as cp
except ImportError:
    cp = None


def is_cupy_device(
    allocator: gtx_allocators.FieldBufferAllocationUtil | None,
) -> bool:
    return gtx_allocators.is_field_allocation_tool_for(allocator, gtx.CUPY_DEVICE_TYPE)


def is_backend_descriptor(
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
) -> TypeGuard[model_backends.BackendDescriptor]:
    if isinstance(backend, dict):
        return all(isinstance(key, str) for key in backend)
    return False


def sync(backend: gtx_typing.Backend | None = None) -> None:
    """
    Synchronize the device if appropriate for the given backend.

    Note: this is and ad-hoc interface, maybe the function should get the device to sync for.
    """
    if backend is not None and is_cupy_device(backend.allocator):
        cp.cuda.runtime.deviceSynchronize()


def synchronized_function(func: Callable[..., Any], *, backend: gtx_typing.Backend | None):
    """
    Wraps a function and synchronizes after execution
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        sync(backend=backend)
        return result

    return wrapper


def synchronized(
    backend: gtx_typing.Backend | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that synchronizes the device after the function execution.
    """
    return functools.partial(synchronized_function, backend=backend)

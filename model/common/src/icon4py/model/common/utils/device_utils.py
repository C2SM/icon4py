# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import gt4py.next.custom_layout_allocators as gtx_allocators
import gt4py.next.typing as gtx_typing


log = logging.getLogger(__name__)

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None


def _warn_if_cub_accelerator_disabled() -> None:
    """
    Warn if CuPy runs without the CUB accelerator, which makes its reductions slow.

    CuPy enables CUB by default on CUDA but no accelerators on ROCm, so the
    default value of 'CUPY_ACCELERATORS' depends on the device.
    """
    if gtx.CUPY_DEVICE_TYPE is None:
        return
    default = "cub" if gtx.CUPY_DEVICE_TYPE == gtx.DeviceType.CUDA else ""
    if "cub" not in os.environ.get("CUPY_ACCELERATORS", default).split(","):
        log.warning(
            'CUPY_ACCELERATORS does not include "cub", so CuPy reductions on GPUs have '
            "degraded performance (on ROCm, CuPy enables no accelerators by default). "
            "Set CUPY_ACCELERATORS=cub."
        )


_warn_if_cub_accelerator_disabled()


def is_cupy_device(allocator: gtx_typing.Allocator | None) -> bool:
    if allocator is None:
        return False

    if gtx.CUPY_DEVICE_TYPE is None:
        return False

    return gtx_allocators.is_field_allocation_tool_for(allocator, gtx.CUPY_DEVICE_TYPE)  # type: ignore [type-var] #gt4py-related typing


def sync(allocator: gtx_typing.Allocator | None = None) -> None:
    """
    Synchronize the device if appropriate for the given backend.

    Note: this is and ad-hoc interface, maybe the function should get the device to sync for.
    """
    # Type annotation already describes that only these types are allowed, but mypy coverage is not great.
    # The explicit assert avoids critical mistakes in using this function.
    assert allocator is None or gtx_allocators.is_field_allocation_tool(allocator)
    if allocator is not None and is_cupy_device(allocator):
        cp.cuda.runtime.deviceSynchronize()


def synchronized_function[**P, R](
    func: Callable[P, R], *, allocator: gtx_typing.Allocator | None
) -> Callable[P, R]:
    """
    Wraps a function and synchronizes after execution
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        result = func(*args, **kwargs)
        sync(allocator=allocator)
        return result

    return wrapper


def synchronized(
    allocator: gtx_typing.Allocator | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that synchronizes the device after the function execution.
    """
    return functools.partial(synchronized_function, allocator=allocator)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
from typing import Any, Final, TypeAlias, TypeGuard

import gt4py.next as gtx
import gt4py.next.custom_layout_allocators as gtx_allocators
import gt4py.next.typing as gtx_typing
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners import dace as gtx_dace, gtfn
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


# DeviceType should always be imported from here, as we might replace it by an ICON4Py internal implementation
DeviceType: TypeAlias = gtx.DeviceType  # noqa: UP040 used with isinstance()
CPU = DeviceType.CPU
GPU = gtx.CUPY_DEVICE_TYPE

type BackendDescriptor = dict[str, Any]
type BackendLike = DeviceType | gtx_typing.Backend | BackendDescriptor | None


DEFAULT_BACKEND: Final = "embedded"

_WORKSPACE_SIZE: Final[int] = (
    200 * 1024 * 1024
)  # 200 MB, this is the default workspace size for ICON4Py programs


def _get_workspace_memory(
    size: int, device_type: gtx.DeviceType, *, backend_device: gtx.DeviceType
) -> Any:
    """
    Returns a workspace memory allocation for the given device type.

    Args:
        size: The size of the workspace memory to allocate.
        device_type: The device type for which the workspace memory is allocated.
        backend_device: The device type of the backend to use for allocation.

    Returns:
        A workspace memory allocation for the given device type.

    Raises:
        ValueError: If the backend cannot allocate memory for the given device type.

    Note that the workspace memory is allocated only once and reused for subsequent calls.
    """
    dim = gtx.Dimension("x")
    allocator = get_allocator(backend_device)
    if not gtx_allocators.is_field_allocation_tool_for(allocator, device_type):
        raise ValueError(f"Backend cannot allocate memory for device type {device_type}")
    if size > _WORKSPACE_SIZE:
        raise ValueError(
            f"Requested workspace size {size} exceeds the maximum allowed {_WORKSPACE_SIZE}"
        )
    if _get_workspace_memory.value is None:
        _get_workspace_memory.value = gtx.constructors.zeros(
            {dim: (0, (_WORKSPACE_SIZE + 7) // 8)}, dtype=gtx.uint64, allocator=allocator
        )

    return _get_workspace_memory.value


_get_workspace_memory.value = None


def is_backend_descriptor(
    backend: BackendLike,
) -> TypeGuard[BackendDescriptor]:
    if isinstance(backend, dict):
        return all(isinstance(key, str) for key in backend)
    return False


def is_cpu_backend(
    backend: BackendLike,
) -> bool:
    if isinstance(backend, gtx_backend.Backend):
        return backend.allocator.device_type == CPU
    return get_allocator(backend).device_type == CPU


def is_gpu_backend(
    backend: BackendLike,
) -> bool:
    if isinstance(backend, gtx_backend.Backend):
        return backend.allocator.device_type == GPU
    return get_allocator(backend).device_type == GPU


def get_allocator(
    backend: BackendLike,
) -> gtx_typing.Backend:
    if isinstance(backend, gtx_backend.Backend):
        return backend
    if backend is None:
        # TODO(havogt): currently the testing infrastructure doesn't allow to specify
        # embedded backend aka `None` for cupy, as there is no separation
        # of allocator and backend.
        return gtx_allocators.device_allocators[CPU]

    if is_backend_descriptor(backend):
        backend = backend["device"]
    if isinstance(backend, gtx.DeviceType):
        return gtx_allocators.device_allocators[backend]
    raise ValueError(f"Cannot get allocator from {backend}")


def make_custom_gtfn_backend(device: DeviceType, **_) -> gtx_typing.Backend:
    on_gpu = device == GPU
    return gtfn.GTFNBackendFactory(gpu=on_gpu)


def make_custom_dace_backend(
    *,
    device: DeviceType,
    auto_optimize: bool = True,
    async_sdfg_call: bool = True,
    optimization_args: dict[str, Any] | None = None,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
    **_,
) -> gtx_typing.Backend:
    """Customize the dace backend with the given configuration parameters.

    Args:
        device: The target device.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU to allow overlapping
            of GPU kernel execution with the Python driver code.
        optimization_args: A `dict` containing configuration parameters for
            the SDFG auto-optimize pipeline.
        use_metrics: Add SDFG instrumentation to collect the metric for stencil
            compute time.
        use_max_domain_range_on_unstructured_shift: When True, compute `as_fieldop`
            expressions everywhere. Otherwise, when all connectivities are given
            at compile time, infer the minimal domain of all `as_fieldop` statically.

    Returns:
        A dace backend with custom configuration for the target device.
    """
    # Use external workspace memory for all programs
    external_memory_allocator = functools.partial(_get_workspace_memory, backend_device=device)
    if optimization_args is None:
        optimization_args = {
            "transient_memory_mode": gtx_transformations.TransientMemoryMode.EXTERNAL,
        }
    else:
        optimization_args["transient_memory_mode"] = (
            gtx_transformations.TransientMemoryMode.EXTERNAL
        )

    on_gpu = device == GPU
    return gtx_dace.make_dace_backend(
        gpu=on_gpu,
        auto_optimize=auto_optimize,
        async_sdfg_call=async_sdfg_call,
        external_memory_allocator=external_memory_allocator,
        optimization_args=optimization_args,
        unstructured_horizontal_has_unit_stride=True,
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )


BACKENDS: dict[str, BackendLike] = {
    "embedded": None,
    "gtfn_cpu": {"backend_factory": make_custom_gtfn_backend, "device": CPU},
    "gtfn_gpu": {"backend_factory": make_custom_gtfn_backend, "device": GPU},
    "dace_cpu": {"backend_factory": make_custom_dace_backend, "device": CPU},
    "dace_gpu": {"backend_factory": make_custom_dace_backend, "device": GPU},
    "cpu": {"device": CPU},
    "gpu": {"device": GPU},
}

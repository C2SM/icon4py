# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import typing

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import allocators as gtx_allocators, backend as gtx_backend
from gt4py.next.program_processors.runners import dace as gtx_dace, gtfn


DEFAULT_BACKEND: typing.Final = "embedded"


# DeviceType should always be imported from here, as we might replace it by an ICON4Py internal implementation
DeviceType: typing.TypeAlias = gtx.DeviceType
CPU = DeviceType.CPU
GPU = gtx.CUPY_DEVICE_TYPE

BackendDescriptor: typing.TypeAlias = dict[str, typing.Any]


def is_backend_descriptor(
    backend: gtx_typing.Backend | DeviceType | BackendDescriptor | None,
) -> typing.TypeGuard[BackendDescriptor]:
    if isinstance(backend, dict):
        return all(isinstance(key, str) for key in backend)
    return False


def get_allocator(
    backend: gtx_typing.Backend | DeviceType | BackendDescriptor | None,
) -> gtx_typing.Backend | None:
    if backend is None or isinstance(backend, gtx_backend.Backend):
        return backend
    if is_backend_descriptor(backend):
        backend = backend["device"]
    if isinstance(backend, DeviceType):
        return gtx_allocators.device_allocators[backend]
    raise ValueError(f"Cannot get allocator from {backend}")


def make_custom_dace_backend(
    device: DeviceType,
    cached: bool = True,
    auto_optimize: bool = True,
    async_sdfg_call: bool = True,
    optimization_args: dict[str, typing.Any] | None = None,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
    **_,
) -> gtx_typing.Backend:
    """Customize the dace backend with the given configuration parameters.

    Args:
        device: The target device.
        cached: Cache the lowered SDFG as a JSON file and the compiled programs.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU to allow overlapping
            of GPU kernel execution with the Python driver code.
        optimization_args: A `dict` containing configuration parameters for
            the SDFG auto-optimize pipeline.
        use_metrics: Add SDFG instrumentation to collect the metric for stencil
            compute time.

    Returns:
        A dace backend with custom configuration for the target device.
    """
    on_gpu = device == GPU
    return gtx_dace.make_dace_backend(
        gpu=on_gpu,
        cached=cached,
        auto_optimize=auto_optimize,
        async_sdfg_call=async_sdfg_call,
        optimization_args=optimization_args,
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
    )


def make_custom_gtfn_backend(device: DeviceType, cached: bool = True, **_) -> gtx_typing.Backend:
    on_gpu = device == GPU
    return gtfn.GTFNBackendFactory(
        gpu=on_gpu,
        cached=cached,
        otf_workflow__cached_translation=cached,
    )


BACKENDS: dict[str, gtx_typing.Backend | None] = {
    "embedded": None,
    "roundtrip": gtx.itir_python,
    "gtfn_cpu": gtx.gtfn_cpu,
    "gtfn_gpu": gtx.gtfn_gpu,
    "dace_cpu": make_custom_dace_backend(device=CPU),
    "dace_gpu": make_custom_dace_backend(device=GPU),
}

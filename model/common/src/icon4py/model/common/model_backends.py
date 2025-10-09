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
from gt4py.next import (
    allocators as gtx_allocators,
    backend as gtx_backend,
    common as gtx_common,
    config as gtx_config,
)
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory


DEFAULT_BACKEND: typing.Final = "embedded"

BACKENDS: dict[str, gtx_typing.Backend | None] = {
    "embedded": None,
    "roundtrip": gtx.itir_python,
    "gtfn_cpu": gtx.gtfn_cpu,
    "gtfn_gpu": gtx.gtfn_gpu,
}

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


try:
    from gt4py.next.program_processors.runners.dace import (
        DaCeBackendFactory,
        GT4PyAutoOptHook,
        GT4PyAutoOptHookFun,
    )

    def make_custom_dace_backend(
        device: DeviceType,
        cached: bool = True,
        auto_optimize: bool = True,
        async_sdfg_call: bool = True,
        optimization_args: dict[str, typing.Any] | None = None,
        optimization_hooks: dict[GT4PyAutoOptHook, GT4PyAutoOptHookFun] | None = None,
        use_memory_pool: bool = True,
        use_metrics: bool = True,
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
            optimization_hooks: A `dict` containing the hooks that should be called,
                in the SDFG auto-optimize pipeline. Only applicable when `auto_optimize=True`.
            use_memory_pool: Allocate temporaries in memory pool, currently only
                supported for GPU (based on CUDA memory pool).
            use_metrics: Add SDFG instrumentation to collect the metric for stencil
                compute time.

        Returns:
            A dace backend with custom configuration for the target device.
        """
        on_gpu = device == GPU
        fixed_optimization_args: typing.Final[dict[str, typing.Any]] = {
            "assume_pointwise": True,
            "gpu_memory_pool": (use_memory_pool if on_gpu else False),
            "optimization_hooks": optimization_hooks,
            "unit_strides_kind": (
                gtx_common.DimensionKind.HORIZONTAL
                if gtx_config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                else None  # let `gt_auto_optimize` select `unit_strides_kind` based on `gpu` argument
            ),
            "validate": False,
            "validate_all": False,
        }

        if optimization_hooks and not auto_optimize:
            raise ValueError("Optimizations hook given, but auto-optimize pipeline is disabled.")
        if optimization_args and not auto_optimize:
            raise ValueError("Optimizations args given, but auto-optimize pipeline is disabled.")
        if optimization_args is None:
            optimization_args = {}
        elif any(arg in fixed_optimization_args for arg in optimization_args):
            raise ValueError(
                f"The following arguments cannot be overriden: {set(optimization_args.keys()).intersection(fixed_optimization_args.keys())}."
            )

        return DaCeBackendFactory(  # type: ignore[return-value] # factory-boy typing not precise enough
            gpu=on_gpu,
            cached=cached,
            auto_optimize=auto_optimize,
            otf_workflow__cached_translation=cached,
            otf_workflow__bare_translation__async_sdfg_call=(async_sdfg_call if on_gpu else False),
            otf_workflow__bare_translation__auto_optimize_args=(
                optimization_args | fixed_optimization_args
            ),
            otf_workflow__bare_translation__use_metrics=use_metrics,
        )

    BACKENDS.update(
        {
            "dace_cpu": make_custom_dace_backend(device=CPU),
            "dace_gpu": make_custom_dace_backend(device=GPU),
        }
    )

except ImportError:
    # dace module not installed, thus the dace backends are not available
    def make_custom_dace_backend(device: DeviceType, **options) -> gtx_typing.Backend:
        raise NotImplementedError("Depends on dace module, which is not installed.")


def make_custom_gtfn_backend(device: DeviceType, cached: bool = True, **_) -> gtx_typing.Backend:
    on_gpu = device == GPU
    return GTFNBackendFactory(
        gpu=on_gpu,
        cached=cached,
        otf_workflow__cached_translation=cached,
    )

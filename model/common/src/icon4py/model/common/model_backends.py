# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
from typing import Any, Final, TypeAlias

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory

from icon4py.model.common import dimension as dims


DEFAULT_BACKEND: Final = "embedded"

BACKENDS: dict[str, gtx_typing.Backend | None] = {
    "embedded": None,
    "roundtrip": gtx.itir_python,
    "gtfn_cpu": gtx.gtfn_cpu,
    "gtfn_gpu": gtx.gtfn_gpu,
}
BackendDescription: TypeAlias = dict[str, Any]


class DeviceType(enum.Enum):
    """
    Type of device: either CPU or GPU
    """

    CPU = gtx.DeviceType.CPU
    GPU = gtx.CUPY_DEVICE_TYPE


try:
    from gt4py.next.program_processors.runners.dace import make_dace_backend

    def make_custom_dace_backend(
        device: str, auto_optimize: bool = True, cached: bool = True, **options
    ) -> gtx_typing.Backend:
        """Customize the dace backend with the following configuration.

        async_sdfg_call:
            In icon4py we want to make an asynchronous SDFG call on gpu to allow
            overlapping of gpu kernel execution with the Python driver code.
        blocking_dim:
            Apply loop-blocking on the vertical dimension `KDim`.
        make_persistent:
            Allocate temporary arrays at SDFG initialization, when it is loaded
            from the binary library. The memory will be persistent across all SDFG
            calls and released only at application exit.
        use_memory_pool:
            Allocate temporaries in memory pool, currently only supported for GPU
            (based on CUDA memory pool).
        use_zero_origin:
            When set to `True`, the SDFG lowering will not generate the start symbol
            of the field range. Select this option if all fields have zero origin.

        Args:
            gpu: Specify if the target device is GPU.
            enable_loop_blocking: Flag to enable loop-blocking transformation on
                the vertical dimension, default `False`.

        Returns:
            A dace backend with custom configuration for the target device.
        """
        on_gpu = device == "gpu"
        return make_dace_backend(
            auto_optimize=auto_optimize,
            cached=cached,
            gpu=on_gpu,
            async_sdfg_call=True,
            blocking_dim=dims.KDim,
            blocking_size=10,
            make_persistent=False,
            use_memory_pool=on_gpu,
            use_zero_origin=True,
        )

    BACKENDS.update(
        {
            "dace_cpu": make_custom_dace_backend(gpu="cpu"),
            "dace_gpu": make_custom_dace_backend(gpu="gpu"),
        }
    )

except ImportError:
    # dace module not installed, thus the dace backends are not available
    def make_custom_dace_backend(gpu: bool) -> gtx_typing.Backend:
        raise NotImplementedError("Depends on dace module, which is not installed.")


def make_custom_gtfn_backend(device: str, cached: bool = True, **options) -> GTFNBackendFactory:
    on_gpu = device == "gpu"
    return GTFNBackendFactory(
        gpu=on_gpu,
        cached=cached,
    )

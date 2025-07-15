# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

import gt4py.next.backend as gtx_backend
from gt4py.next import gtfn_cpu, gtfn_gpu, itir_python

from icon4py.model.common import dimension as dims


DEFAULT_BACKEND: Final = "embedded"

BACKENDS: dict[str, gtx_backend.Backend | None] = {
    "embedded": None,
    "roundtrip": itir_python,
    "gtfn_cpu": gtfn_cpu,
    "gtfn_gpu": gtfn_gpu,
}


try:
    from gt4py.next.program_processors.runners.dace import make_dace_backend

    def make_custom_dace_backend(
        gpu: bool, enable_loop_blocking: bool = False
    ) -> gtx_backend.Backend:
        """Customize the dace backend with the following configuration.

        async_sdfg_call:
            In icon4py we want to make an asynchronous SDFG call on gpu to allow
            overlapping of gpu kernel execution with the Python driver code.
        blocking_dim:
            Apply loop-blocking on the vertical dimension `KDim`, if the input
            argument `enable_loop_blocking` is `True`.
        use_memory_pool:
            Use CUDA memory pool for GPU backend.
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
        return make_dace_backend(
            auto_optimize=True,
            cached=True,
            gpu=gpu,
            async_sdfg_call=False,
            blocking_dim=(dims.KDim if enable_loop_blocking else None),
            blocking_size=10,
            make_persistent=False,
            use_memory_pool=False,
            use_zero_origin=False,
        )

    BACKENDS.update(
        {
            "dace_cpu": make_custom_dace_backend(gpu=False),
            "dace_gpu": make_custom_dace_backend(gpu=True),
        }
    )

except ImportError:
    # dace module not installed, ignore dace backends
    pass

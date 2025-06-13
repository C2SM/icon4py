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

    # In icon4py we want to make an asynchronous SDFG call on gpu to allow overlapping
    #   of gpu kernel execution with the Python driver code (same behavior as in GTFN).
    # Besides, it is safe to assume that the field layout does not change
    #   between multiple calls to a gt4py program, therefore we can make temporary
    #   arrays persistent (thus, allocated at SDFG initialization) and we do not
    #   need to update the array shape and strides on each SDFG call.
    # We also enable loop-blocking on the vertical dimension for gpu target.
    def make_custom_dace_backend(gpu: bool) -> gtx_backend.Backend:
        return make_dace_backend(
            auto_optimize=True,
            cached=True,
            gpu=gpu,
            async_sdfg_call=True,
            make_persistent=True,
            blocking_dim=(dims.KDim if gpu else None),
            blocking_size=10,
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

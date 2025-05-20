# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
from typing import Final

import gt4py.next.backend as gtx_backend
from gt4py.next import gtfn_cpu, gtfn_gpu, itir_python

from icon4py.model.common.utils import data_allocation as data_alloc


DEFAULT_BACKEND: Final = "embedded"

BACKENDS: dict[str, gtx_backend.Backend | None] = {
    "embedded": None,
    "roundtrip": itir_python,
    "gtfn_cpu": gtfn_cpu,
    "gtfn_gpu": gtfn_gpu,
}


def _customize_dace_backend(dace_backend: gtx_backend.Backend) -> gtx_backend.Backend:
    # In icon4py it is safe to assume that the field layout does not change
    #   between multiple calls to a gt4py program, therefore we can make temporary
    #   array persistent (thus, allocated at SDFG initialization) and we do not
    #   need to update array shape and strides on each SDFG call.
    # Besides, we want to make an asynchronous SDFG call on gpu to allow overlapping
    #   of gpu kernel execution with the Python driver code (same behavior as in GTFN).
    return dataclasses.replace(
        dace_backend,
        executor=dataclasses.replace(
            dace_backend.executor.step,
            translation=dataclasses.replace(
                dace_backend.executor.step.translation.step,
                make_persistent=True,
                async_sdfg_call=data_alloc.is_cupy_device(dace_backend),
            ),
            bindings=functools.partial(
                dace_backend.executor.step.bindings,
                make_persistent=True,
            ),
        ),
    )


try:
    from gt4py.next.program_processors.runners.dace import (
        run_dace_cpu_cached as run_dace_cpu,
        run_dace_gpu_cached as run_dace_gpu,
    )

    BACKENDS.update(
        {
            "dace_cpu": _customize_dace_backend(run_dace_cpu),
            "dace_gpu": _customize_dace_backend(run_dace_gpu),
        }
    )

except ImportError:
    # dace module not installed, ignore dace backends
    pass

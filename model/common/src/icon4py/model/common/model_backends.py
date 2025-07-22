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


DEFAULT_BACKEND: Final = "embedded"

BACKENDS: dict[str, gtx_backend.Backend | None] = {
    "embedded": None,
    "roundtrip": itir_python,
    "gtfn_cpu": gtfn_cpu,
    "gtfn_gpu": gtfn_gpu,
}

try:
    from gt4py.next.program_processors.runners.dace import (
        run_dace_cpu_cached as run_dace_cpu,
        run_dace_gpu_cached as run_dace_gpu,
    )

    BACKENDS.update(
        {
            "dace_cpu": run_dace_cpu,
            "dace_gpu": run_dace_gpu,
        }
    )

except ImportError:
    # dace module not installed, ignore dace backends
    pass

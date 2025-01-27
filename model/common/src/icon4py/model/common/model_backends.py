# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Final

from gt4py.next import gtfn_cpu, gtfn_gpu, itir_python


DEFAULT_BACKEND: Final = "roundtrip"

BACKENDS: dict[str, Callable] = {
    "embedded": None,
    "roundtrip": itir_python,
    "gtfn_cpu": gtfn_cpu,
    "gtfn_gpu": gtfn_gpu,
}

try:
    from gt4py.next.program_processors.runners.dace import (
        run_dace_cpu,
        run_dace_cpu_noopt,
        run_dace_gpu,
        run_dace_gpu_noopt,
    )

    BACKENDS.update(
        {
            "dace_cpu": run_dace_cpu,
            "dace_gpu": run_dace_gpu,
            "dace_cpu_noopt": run_dace_cpu_noopt,
            "dace_gpu_noopt": run_dace_gpu_noopt,
        }
    )

except ImportError:
    # dace module not installed, ignore dace backends
    pass

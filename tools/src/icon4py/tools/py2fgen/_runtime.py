# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import time as _time


try:
    import cupy as cp

    def _sync():
        cp.cuda.runtime.deviceSynchronize()
except ImportError:
    cp = None

    def _sync():
        pass


def perf_counter():
    _sync()
    return _time.perf_counter()


def get_cupy_info():
    try:
        import cupy as cp

        return cp.show_config()
    except ImportError:
        return "CuPy is not installed"

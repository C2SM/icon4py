# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import time as _time

from icon4py.tools.py2fgen import runtime_config


try:
    import cupy as cp  # type: ignore[import-not-found]

    def device_synchronize() -> None:
        cp.cuda.runtime.deviceSynchronize()
except ImportError:
    cp = None

    def device_synchronize() -> None:
        pass


def gpu_stream(external_gpu_stream: int) -> contextlib.AbstractContextManager:
    """Make the external GPU stream, if provided, the current CuPy stream in the context."""
    if cp is None or external_gpu_stream < 0:  # no external GPU stream
        return contextlib.nullcontext()
    # `ExternalStream` does not take ownership of the stream.
    return cp.cuda.ExternalStream(external_gpu_stream)


def perf_counter() -> float:
    device_synchronize()
    return _time.perf_counter()


if runtime_config.USE_DEVICE and cp is None:
    raise ImportError("'PY2FGEN_USE_DEVICE' is enabled, but 'cupy' is not installed.")


def use_device(device_enabled: bool) -> bool:
    if runtime_config.USE_DEVICE is None:
        return bool(device_enabled)
    if device_enabled and not runtime_config.USE_DEVICE:
        raise RuntimeError(
            "'PY2FGEN_USE_DEVICE' is disabled, but the Fortran side passes device pointers."
        )
    return runtime_config.USE_DEVICE

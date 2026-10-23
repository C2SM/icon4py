# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Diagnostics of the icon4py ComIn plugin's timing mode (ICON4PY_COMIN_TIMING=1): the state of the
ICON process, read from Linux '/proc' and the CUDA driver. Information only; nothing here raises.
"""

import ctypes
import os
import pathlib
import resource
from typing import Final


_CU_CTX_SCHED: Final = {0: "auto", 1: "spin", 2: "yield", 4: "blocking-sync"}


def cuda_context_flags() -> str:
    """Scheduling flags of device 0's primary CUDA context (driver API)."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        device, flags, active = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return "unavailable"
        if cuda.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)) != 0:
            return "unavailable"
        sched = _CU_CTX_SCHED.get(flags.value & 0x7, hex(flags.value & 0x7))
        return f"flags {flags.value:#x} (schedule {sched}), active {active.value}"
    except OSError:
        return "unavailable (no libcuda)"


def process_state() -> str:
    """
    Native threads of this process with their CPU time, child processes, and the calling
    thread's CPU affinity and context switches (Linux '/proc'; information only, never raises).
    """
    try:
        tick = os.sysconf("SC_CLK_TCK")
        threads, children = [], []
        for task in sorted(pathlib.Path("/proc/self/task").iterdir(), key=lambda t: int(t.name)):
            try:
                name = (task / "comm").read_text().strip()
                fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
                cpu = (int(fields[11]) + int(fields[12])) / tick  # utime + stime
                children += (task / "children").read_text().split()
            except OSError:
                continue
            threads.append(f"{name}:{cpu:.2f}")
        affinity = os.sched_getaffinity(0)
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        return (
            f"{len(threads)} native threads [name:CPU s] {' '.join(threads)}; {len(children)} child"
            f" processes; this thread: {len(affinity)} CPUs allowed ({min(affinity)}-{max(affinity)}),"
            f" {usage.ru_nvcsw} voluntary and {usage.ru_nivcsw} involuntary context switches;"
            f" primary CUDA context {cuda_context_flags()}"
        )
    except Exception as error:
        return f"unavailable ({error!r})"

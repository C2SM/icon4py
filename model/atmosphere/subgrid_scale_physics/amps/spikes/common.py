# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for M0 feasibility spikes. Not part of the amps package API."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import gt4py.next as gtx
import numpy as np
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached

from icon4py.model.common import dimension as dims


NCELLS = 4096
NLEV = 61
GENDIR = pathlib.Path(__file__).parent / "_generated"


def backends() -> dict:
    return {"embedded": None, "gtfn_cpu": run_gtfn_cached}


def make_field(array: np.ndarray) -> gtx.Field:
    dim_map = {2: (dims.CellDim, dims.KDim), 1: (dims.KDim,)}
    return gtx.as_field(dim_map[array.ndim], array)


def zeros_field(shape: tuple[int, ...] = (NCELLS, NLEV)) -> gtx.Field:
    return make_field(np.zeros(shape))


def time_first_and_steady(fn, n_steady: int = 10) -> tuple[float, float]:
    """Return (first_call_seconds, steady_state_seconds). First call includes
    toolchain compilation for compiled backends."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n_steady):
        fn()
    steady = (time.perf_counter() - t0) / n_steady
    return first, steady


def load_generated_operator(source: str, module_name: str, attr: str):
    """Write generated DSL source to a real file (gt4py parses inspect.getsource)
    and import the named attribute from it."""
    GENDIR.mkdir(exist_ok=True)
    path = GENDIR / f"{module_name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, attr)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Small standalone utilities. `recursion_limit` below is amps' OWN copy
(not an import) of the identical helper already precedented in this repo
by `icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver.utils.
recursion_limit` (docs/superpowers/facts/m1/icon4py-m1-conventions.md
"F5" SS6, quoting that file verbatim) -- copied for provenance, not
reused via import, so amps carries no runtime dependency on muphys (M0
carry-forward #6)."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator


@contextlib.contextmanager
def recursion_limit(limit: int) -> Generator[None, None, None]:
    """Temporarily raise Python's recursion limit, restoring the
    original value on exit (even if the body raises).

    gt4py 1.1.11's ITIR transform pipeline (the `eve`-framework tree
    visitors, e.g. `MergeLet`) recurses per-node and can exceed Python's
    default recursion limit (1000) when compiling a large generated
    expression tree on `gtfn_cpu` -- see
    `spikes/spike_b_collection_codegen.py` (F5 SS6): its 40-bin collection
    kernel hits `RecursionError` without this raised, and compiles
    (slowly) with it raised to `10**5`. Use as:

        with recursion_limit(10**5):
            program.compile(...)

    Copied verbatim from muphys's own `driver/utils.py::recursion_limit`
    (identical body: same `sys.getrecursionlimit()`/`sys.setrecursionlimit()`
    pair, same try/finally) -- see this module's docstring for why it is
    a copy, not an import.
    """
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(original_limit)

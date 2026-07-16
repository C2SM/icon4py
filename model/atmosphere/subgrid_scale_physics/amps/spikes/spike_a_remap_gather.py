# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Spike A: gather-formulated bin remap and value-indexed table lookup via
as_offset. Answers: does as_offset(Koff, int_field) work on embedded and
gtfn_cpu, for (a) field self-gather with a computed shift, (b) a K-stored
table gathered at an index computed from field values?

Follow-up (same task): the plain K-only-table idiom below is a NO-GO (see the
comment block under `_table_gather`, kept for the record). Two rescue variants
were probed: broadcasting the table to (Cell, K) before the gather (still a
NO-GO, differently on each backend) and passing the table pre-tiled to full
(Cell, K) shape, i.e. memory-replicated across cells (a GO on both backends).

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_a_remap_gather.py
"""

from __future__ import annotations

import common
import gt4py.next as gtx
import numpy as np
from gt4py.next import astype, broadcast, maximum, minimum
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


@gtx.field_operator
def _gather_shift(
    q: fa.CellKField[ta.wpfloat], koffset: fa.CellKField[gtx.int32]
) -> fa.CellKField[ta.wpfloat]:
    return q(as_offset(Koff, koffset))


# NOT EXECUTED -- kept as a comment for the record (a spike finding, not code).
# The straightforward table-gather idiom below is a hard NO-GO. gt4py 1.1.11's
# foast return-type deduction for `field(as_offset(Off, expr))` derives the
# result dims purely from the *remapped field's own* dims (here `table`, which
# only has KDim) with the offset's source dim swapped for its target dim; it
# does not pick up extra dims carried by the offset expression `idx - k_index`
# (which is (CellDim, KDim) because `idx` comes from the per-cell field `t`). So
# the annotated `CellKField` return type never matches the deduced
# `Field[[K], float64]`, and `@gtx.field_operator` raises DSLError at decoration
# time -- before any backend is even selected, so both embedded and gtfn_cpu
# fail identically. See `_table_gather_broadcast` and `_table_gather_tiled`
# below for the two rescue attempts.
#
# @gtx.field_operator
# def _table_gather(
#     t: fa.CellKField[ta.wpfloat],
#     table: gtx.Field[gtx.Dims[dims.KDim], ta.wpfloat],
#     k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
# ) -> fa.CellKField[ta.wpfloat]:
#     idx = astype(maximum(0.0, minimum(t, 59.0)), gtx.int32)  # noqa: ERA001
#     return table(as_offset(Koff, idx - k_index))  # noqa: ERA001
#
# Verbatim error (identical on embedded and gtfn_cpu, raised at decoration time):
#
# gt4py.next.errors.exceptions.DSLError: Annotated return type does not match
# deduced return type: annotation is 'Field[[Cell, K], float64]', got
# 'Field[[K], float64]'.


@gtx.field_operator
def _table_gather_broadcast(
    t: fa.CellKField[ta.wpfloat],
    table: gtx.Field[gtx.Dims[dims.KDim], ta.wpfloat],
    k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
) -> fa.CellKField[ta.wpfloat]:
    """Rescue attempt 1: broadcast the K-only table to (Cell, K) *before* the
    gather, so the field being remapped already carries CellDim and the
    decoration-time dims mismatch above disappears. Decoration succeeds, but
    execution is a NO-GO on both backends (see run_table_gather_broadcast for
    the verbatim errors) -- broadcast + dynamic as_offset gather in the same
    operator is not supported by gt4py 1.1.11's embedded gather kernel nor its
    gtfn lowering.
    """
    idx = astype(maximum(0.0, minimum(t, 59.0)), gtx.int32)
    table_ck = broadcast(table, (dims.CellDim, dims.KDim))
    return table_ck(as_offset(Koff, idx - k_index))


@gtx.field_operator
def _table_gather_tiled(
    t: fa.CellKField[ta.wpfloat],
    table: fa.CellKField[ta.wpfloat],
    k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
) -> fa.CellKField[ta.wpfloat]:
    """Rescue attempt 2 (memory-replicated fallback): the table is declared and
    passed in as a full (Cell, K) field (values tiled across cells by the
    caller), structurally identical to _gather_shift's self-gather. GO on both
    backends.
    """
    idx = astype(maximum(0.0, minimum(t, 59.0)), gtx.int32)
    return table(as_offset(Koff, idx - k_index))


def run_gather_shift(backend) -> None:
    rng = np.random.default_rng(42)
    q_np = rng.uniform(size=(common.NCELLS, common.NLEV))
    k = np.arange(common.NLEV)
    koffset_np = rng.integers(-k, common.NLEV - 1 - k, size=(common.NCELLS, common.NLEV))
    q = common.make_field(q_np)
    koffset = gtx.as_field((dims.CellDim, dims.KDim), koffset_np.astype(np.int32))
    out = common.zeros_field()
    op = _gather_shift.with_backend(backend) if backend is not None else _gather_shift

    def call():
        # Koff is a Cartesian (K-self) offset with a per-point dynamic shift: it
        # needs no connectivity entry (there is no neighbor table), so the offset
        # provider is empty. A bare `{"Koff": dims.KDim}` is rejected by gt4py
        # 1.1.11's compiled-backend path (see the task report for the verbatim
        # error); an empty dict passes its offset-provider validation vacuously
        # and both backends resolve the shift from the field argument itself.
        op(q, koffset, out=out, offset_provider={})

    first, steady = common.time_first_and_steady(call)
    expected = np.take_along_axis(q_np, k[None, :] + koffset_np, axis=1)
    assert np.allclose(out.asnumpy(), expected), "gather_shift mismatch vs numpy"
    print(
        f"RESULT gather_shift backend={'embedded' if backend is None else 'gtfn_cpu'} "
        f"first={first:.3f}s steady={steady * 1e3:.2f}ms"
    )


def run_table_gather_broadcast(backend) -> bool:
    backend_name = "embedded" if backend is None else "gtfn_cpu"
    rng = np.random.default_rng(43)
    t_np = rng.uniform(0.0, 59.0, size=(common.NCELLS, common.NLEV))
    table_np = np.linspace(100.0, 200.0, common.NLEV)
    t = common.make_field(t_np)
    table = common.make_field(table_np)
    k_index = gtx.as_field((dims.KDim,), np.arange(common.NLEV, dtype=np.int32))
    out = common.zeros_field()
    op = (
        _table_gather_broadcast.with_backend(backend)
        if backend is not None
        else _table_gather_broadcast
    )

    try:
        op(t, table, k_index, out=out, offset_provider={})
    except Exception as exc:
        print(f"RESULT table_gather_broadcast backend={backend_name} FAILED: {exc!r}")
        return False

    idx = np.clip(t_np, 0.0, 59.0).astype(np.int32)
    assert np.allclose(out.asnumpy(), table_np[idx]), "table_gather_broadcast mismatch vs numpy"
    print(f"RESULT table_gather_broadcast backend={backend_name} unexpectedly succeeded")
    return True


def run_table_gather_tiled(backend) -> bool:
    backend_name = "embedded" if backend is None else "gtfn_cpu"
    rng = np.random.default_rng(43)
    t_np = rng.uniform(0.0, 59.0, size=(common.NCELLS, common.NLEV))
    table_np_1d = np.linspace(100.0, 200.0, common.NLEV)
    table_np_tiled = np.tile(table_np_1d, (common.NCELLS, 1))
    t = common.make_field(t_np)
    table = common.make_field(table_np_tiled)
    k_index = gtx.as_field((dims.KDim,), np.arange(common.NLEV, dtype=np.int32))
    out = common.zeros_field()
    op = _table_gather_tiled.with_backend(backend) if backend is not None else _table_gather_tiled

    def call():
        op(t, table, k_index, out=out, offset_provider={})

    first, steady = common.time_first_and_steady(call)
    idx = np.clip(t_np, 0.0, 59.0).astype(np.int32)
    expected = table_np_1d[idx]
    assert np.allclose(out.asnumpy(), expected), "table_gather_tiled mismatch vs numpy"
    print(
        f"RESULT table_gather_tiled backend={backend_name} "
        f"first={first:.3f}s steady={steady * 1e3:.2f}ms"
    )
    return True


if __name__ == "__main__":
    table_gather_broadcast_ok = True
    table_gather_tiled_ok = True
    for backend in common.backends().values():
        run_gather_shift(backend)
        table_gather_broadcast_ok &= run_table_gather_broadcast(backend)
        table_gather_tiled_ok &= run_table_gather_tiled(backend)
    print(
        "SPIKE A: gather_shift is a GO; plain K-only table_gather is a NO-GO "
        "(see comment block above _table_gather_broadcast); table_gather_broadcast "
        f"is a {'GO' if table_gather_broadcast_ok else 'NO-GO'}; table_gather_tiled "
        f"(memory-replicated fallback) is a {'GO' if table_gather_tiled_ok else 'NO-GO'}."
    )

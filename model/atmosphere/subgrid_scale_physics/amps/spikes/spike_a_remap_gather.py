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

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_a_remap_gather.py
"""

from __future__ import annotations

import common
import gt4py.next as gtx
import numpy as np
from gt4py.next import astype, maximum, minimum
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


@gtx.field_operator
def _gather_shift(
    q: fa.CellKField[ta.wpfloat], koffset: fa.CellKField[gtx.int32]
) -> fa.CellKField[ta.wpfloat]:
    return q(as_offset(Koff, koffset))


# NOTE: gt4py 1.1.11's foast return-type deduction for `field(as_offset(Off, expr))`
# derives the result dims purely from the *remapped field's own* dims (here `table`,
# which only has KDim) with the offset's source dim swapped for its target dim; it
# does not pick up extra dims carried by the offset expression `idx - k_index`
# (which is (CellDim, KDim) because `idx` comes from the per-cell field `t`). So the
# annotated `CellKField` return type never matches the deduced `Field[[K], float64]`,
# and `@gtx.field_operator` raises DSLError at decoration time -- before any backend
# is even selected. This is a genuine, backend-independent DSL limitation of the
# table-gather idiom as formulated, not a transcription typo, so we catch it here
# (instead of letting it crash the whole module at import) to keep the independent
# gather_shift measurements runnable, and report it verbatim as a spike result below.
try:

    @gtx.field_operator
    def _table_gather(
        t: fa.CellKField[ta.wpfloat],
        table: gtx.Field[gtx.Dims[dims.KDim], ta.wpfloat],
        k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
    ) -> fa.CellKField[ta.wpfloat]:
        idx = astype(maximum(0.0, minimum(t, 59.0)), gtx.int32)
        return table(as_offset(Koff, idx - k_index))

except Exception as exc:
    _table_gather = None
    _TABLE_GATHER_DSL_ERROR: Exception | None = exc
else:
    _TABLE_GATHER_DSL_ERROR = None


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
        # provider is empty; see the fix note above `_table_gather` for why a bare
        # `{"Koff": dims.KDim}` is rejected by gt4py 1.1.11's compiled-backend path.
        op(q, koffset, out=out, offset_provider={})

    first, steady = common.time_first_and_steady(call)
    expected = np.take_along_axis(q_np, k[None, :] + koffset_np, axis=1)
    assert np.allclose(out.asnumpy(), expected), "gather_shift mismatch vs numpy"
    print(
        f"RESULT gather_shift backend={'embedded' if backend is None else 'gtfn_cpu'} "
        f"first={first:.3f}s steady={steady * 1e3:.2f}ms"
    )


def run_table_gather(backend) -> bool:
    backend_name = "embedded" if backend is None else "gtfn_cpu"
    if _TABLE_GATHER_DSL_ERROR is not None:
        print(
            f"RESULT table_gather backend={backend_name} FAILED "
            f"(DSLError at field_operator decoration, backend-independent): "
            f"{_TABLE_GATHER_DSL_ERROR}"
        )
        return False

    rng = np.random.default_rng(43)
    t_np = rng.uniform(0.0, 59.0, size=(common.NCELLS, common.NLEV))
    table_np = np.linspace(100.0, 200.0, common.NLEV)
    t = common.make_field(t_np)
    table = common.make_field(table_np)
    k_index = gtx.as_field((dims.KDim,), np.arange(common.NLEV, dtype=np.int32))
    out = common.zeros_field()
    op = _table_gather.with_backend(backend) if backend is not None else _table_gather

    def call():
        op(t, table, k_index, out=out, offset_provider={})

    first, steady = common.time_first_and_steady(call)
    idx = np.clip(t_np, 0.0, 59.0).astype(np.int32)
    assert np.allclose(out.asnumpy(), table_np[idx]), "table_gather mismatch vs numpy"
    print(
        f"RESULT table_gather backend={backend_name} first={first:.3f}s steady={steady * 1e3:.2f}ms"
    )
    return True


if __name__ == "__main__":
    table_gather_ok = True
    for backend in common.backends().values():
        run_gather_shift(backend)
        table_gather_ok &= run_table_gather(backend)
    if table_gather_ok:
        print("SPIKE A: PASS")
    else:
        print(
            "SPIKE A: PARTIAL -- gather_shift idiom is a GO; table_gather "
            "(as_offset K-only table lookup) idiom is a NO-GO, see RESULT above"
        )

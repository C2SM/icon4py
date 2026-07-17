# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Spike D: saturation vapor pressure over liquid -- analytic Murphy-Koop in
DSL vs Fortran-style table + linear interpolation (estbar: 150 entries,
T index = int(T)-163, clamped). Decides whether the port replaces the
Fortran LUTs with analytic formulas (spec 4, core/thermo.py).

Fortran reference: estbar(i) tabulates Murphy-Koop at T = 163+i K
(i = 1..150), linear interp between entries; see QSPARM2 in
mod_amps_utility.F90 (scale_amps repo).

Relationship to Spike A (spike_a_remap_gather.py): that spike found the
*plain* K-only-table-gather idiom (`table(as_offset(Koff, expr))` returned
directly, with no further arithmetic) is a decoration-time NO-GO -- FOAST
return-type deduction derives the result dims purely from the remapped
field's own dims (K-only for a KDim-only table), ignoring the extra CellDim
carried by the *offset expression*, so the annotated `CellKField` return
type never matches and `@gtx.field_operator` raises `DSLError` before any
backend runs.

The brief for this spike assumed `_esat_table_k_only` below (structurally:
K-only table, CellK-computed offset) would hit that exact same decoration
failure. It does NOT -- verified by actually attempting it (not assumed from
Spike A's finding). The two cases differ in one relevant way: Spike A's
`_table_gather` *returns the gathered field directly*; `_esat_table_k_only`
instead combines *two* gathered fields (`e0`, `e1`) arithmetically with
`frac` (a genuinely `(Cell, K)`-shaped intermediate, since `frac` derives
from `t`). That extra arithmetic changes which code path FOAST's dims
deduction takes:

  - Decoration: SUCCEEDS (contra the brief's expectation) -- the annotated
    `CellKField` return type is accepted.
  - embedded execution: FAILS at run time (not decoration time), for both
    `offset_provider={}` and `offset_provider={"Koff": dims.KDim}`, with:
      ValueError("Dimensions 'K[vertical], Cell[horizontal]' are not
      ordered correctly, expected 'Cell[horizontal], K[vertical]'.")
    i.e. the deduced dims *do* include both Cell and K (unlike Spike A's
    pure-K-only deduction), but in the wrong order, and gt4py's embedded
    executor enforces canonical (Cell, K) ordering at run time rather than
    at decoration time.
  - gtfn_cpu execution: unexpectedly SUCCEEDS with `offset_provider={}` and
    produces numerically plausible values (max relative error vs analytic
    ~4e-3 at a 16-cell probe scale, consistent with 1 K table resolution) --
    see `run_k_only_variant` below for the full-scale, reproducible number.
    This asymmetry (gtfn_cpu tolerating a dims-order mismatch that embedded
    rejects) was not observable in Spike A, since that spike's K-only case
    never got past decoration on either backend.

Net: the K-only table idiom is *not* a clean decoration-time NO-GO here, but
it is backend-inconsistent (fails on embedded, "works" on gtfn_cpu) and thus
not adopted. `_esat_table_tiled` below -- the memory-replicated fallback
Spike A proved is a GO on both backends -- is the measured, portable path;
its RESULT lines gate this spike's correctness asserts and timings.

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_d_esat.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import common
import gt4py.next as gtx
import numpy as np
from gt4py.next import astype, exp, log, maximum, minimum, tanh
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


NLEV = 200  # table needs >= 150 K-levels for the gather idiom


def gt4py_cache_dir() -> Path:
    """The on-disk gt4py build cache directory used by this spike's run
    command (`cd $WT && uv run ...`), i.e. `<worktree_root>/.gt4py_cache`.
    Copied from spike_b_collection_codegen.py / spike_c_wide_scan.py (spikes
    are standalone scripts).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent / ".gt4py_cache"
    return Path.cwd() / ".gt4py_cache"


def clear_gt4py_cache() -> None:
    """Remove the gt4py on-disk build cache so the next gtfn_cpu compile is a
    genuine cold compile, not served from PERSISTENT on-disk cache left by a
    previous run. Safe: this is a rebuildable compiler-artifact cache; nothing
    else is touched.
    """
    cache_dir = gt4py_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"  (cleared gt4py cache: {cache_dir})")
    else:
        print(f"  (gt4py cache dir did not exist, nothing to clear: {cache_dir})")


def murphy_koop_np(t: np.ndarray) -> np.ndarray:
    return np.exp(
        54.842763
        - 6763.22 / t
        - 4.210 * np.log(t)
        + 0.000367 * t
        + np.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
    )


def esat_table_interp_np(t: np.ndarray, table_1d: np.ndarray) -> np.ndarray:
    """numpy replica of the table-interpolation math shared by
    `_esat_table_k_only` / `_esat_table_tiled` (same clamp/index/frac
    arithmetic). This -- not `murphy_koop_np` -- is the correctness
    reference for the DSL table path: the table path is expected to *deviate
    slightly* from analytic (that deviation is the accuracy finding this
    spike measures), so it must be checked against a faithful replica of its
    own math, at tight tolerance, rather than against analytic.
    """
    ti = np.clip(t - 163.0, 1.0, 149.0)
    i0 = ti.astype(np.int32)
    frac = ti - i0.astype(np.float64)
    e0 = table_1d[i0 - 1]
    e1 = table_1d[i0]
    return e0 + frac * (e1 - e0)


@gtx.field_operator
def _esat_analytic(t: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return exp(
        54.842763
        - 6763.22 / t
        - 4.210 * log(t)
        + 0.000367 * t
        + tanh(0.0415 * (t - 218.8)) * (53.878 - 1331.22 / t - 9.44523 * log(t) + 0.014025 * t)
    )


@gtx.field_operator
def _esat_table_k_only(
    t: fa.CellKField[ta.wpfloat],
    table: gtx.Field[gtx.Dims[dims.KDim], ta.wpfloat],
    k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
) -> fa.CellKField[ta.wpfloat]:
    """Brief's original K-only-table idiom -- see module docstring for the
    measured (not assumed) decoration/execution behavior: decoration
    succeeds, embedded execution fails, gtfn_cpu execution unexpectedly
    succeeds. Not adopted (backend-inconsistent); kept to document the
    finding and to produce the verbatim embedded failure. Not used for any
    gating assert in this spike.
    """
    ti = maximum(1.0, minimum(t - 163.0, 149.0))
    i0 = astype(ti, gtx.int32)
    frac = ti - astype(i0, ta.wpfloat)
    e0 = table(as_offset(Koff, i0 - 1 - k_index))
    e1 = table(as_offset(Koff, i0 - k_index))
    return e0 + frac * (e1 - e0)


@gtx.field_operator
def _esat_table_tiled(
    t: fa.CellKField[ta.wpfloat],
    table: fa.CellKField[ta.wpfloat],
    k_index: gtx.Field[gtx.Dims[dims.KDim], gtx.int32],
) -> fa.CellKField[ta.wpfloat]:
    """Portable table-interp idiom: the table is passed in as a full
    (Cell, K) field (values memory-replicated/tiled across cells by the
    caller), matching Spike A's proven `_table_gather_tiled` rescue. GO on
    both backends -- this is the measured path for this spike.
    """
    ti = maximum(1.0, minimum(t - 163.0, 149.0))
    i0 = astype(ti, gtx.int32)
    frac = ti - astype(i0, ta.wpfloat)
    e0 = table(as_offset(Koff, i0 - 1 - k_index))
    e1 = table(as_offset(Koff, i0 - k_index))
    return e0 + frac * (e1 - e0)


def run_k_only_variant(
    backend_name: str,
    backend,
    *,
    t: gtx.Field,
    table: gtx.Field,
    k_index: gtx.Field,
    exact: np.ndarray,
) -> None:
    """Non-gating: documents the backend-inconsistent K-only idiom's actual
    run-time behavior (see module docstring). A failure here is itself a
    valid, expected spike datum (mirrors spike_a_remap_gather.py's try/except
    pattern for the same reason), not an error in this script.
    """
    out = gtx.as_field((dims.CellDim, dims.KDim), np.zeros((t.shape[0], t.shape[1])))
    op = _esat_table_k_only.with_backend(backend) if backend is not None else _esat_table_k_only

    def call() -> None:
        op(t, table, k_index, out=out, offset_provider={})

    try:
        first, steady = common.time_first_and_steady(call)
    except Exception as exc:  # a backend failure IS the datum here
        print(f"RESULT esat_table_k_only backend={backend_name} FAILED: {exc!r}")
        return

    got = out.asnumpy()
    rel = np.abs(got - exact) / exact
    print(
        f"RESULT esat_table_k_only backend={backend_name} first={first:.2f}s "
        f"steady={steady * 1e3:.2f}ms max_rel_err_vs_analytic={rel.max():.2e} "
        "(non-portable idiom, documented not adopted -- see module docstring)"
    )


def main() -> None:
    rng = np.random.default_rng(5)
    t_np = rng.uniform(180.0, 310.0, size=(common.NCELLS, NLEV))
    t = gtx.as_field((dims.CellDim, dims.KDim), t_np)

    table_np = np.zeros(NLEV)
    table_np[:150] = murphy_koop_np(np.arange(1, 151) + 163.0)
    table_k_only = gtx.as_field((dims.KDim,), table_np)
    table_tiled = gtx.as_field((dims.CellDim, dims.KDim), np.tile(table_np, (common.NCELLS, 1)))
    k_index = gtx.as_field((dims.KDim,), np.arange(NLEV, dtype=np.int32))

    exact = murphy_koop_np(t_np)
    table_interp_exact = esat_table_interp_np(t_np, table_np)

    for name in ("gtfn_cpu", "embedded"):
        backend = common.backends()[name]
        if name == "gtfn_cpu":
            clear_gt4py_cache()

        out_a = gtx.as_field((dims.CellDim, dims.KDim), np.zeros_like(t_np))
        op_a = _esat_analytic.with_backend(backend) if backend is not None else _esat_analytic
        first_a, steady_a = common.time_first_and_steady(
            lambda op_a=op_a, out_a=out_a: op_a(t, out=out_a, offset_provider={})
        )
        assert np.allclose(out_a.asnumpy(), exact, rtol=1e-12), (
            f"esat_analytic {name} does not match numpy Murphy-Koop at rtol=1e-12"
        )
        print(
            f"RESULT esat_analytic backend={name} first={first_a:.2f}s "
            f"steady={steady_a * 1e3:.2f}ms"
        )

        out_t = gtx.as_field((dims.CellDim, dims.KDim), np.zeros_like(t_np))
        op_t = _esat_table_tiled.with_backend(backend) if backend is not None else _esat_table_tiled
        first_t, steady_t = common.time_first_and_steady(
            lambda op_t=op_t, out_t=out_t: op_t(
                t, table_tiled, k_index, out=out_t, offset_provider={}
            )
        )
        got_t = out_t.asnumpy()
        # Correctness: checked against the numpy replica of the TABLE'S OWN
        # interpolation math (esat_table_interp_np), at rtol=1e-12 -- not
        # against analytic, since the table path is *expected* to deviate
        # from analytic (that deviation is the accuracy finding, reported
        # below via max_rel_err_vs_analytic, un-asserted).
        assert np.allclose(got_t, table_interp_exact, rtol=1e-12), (
            f"esat_table_tiled {name} does not match its own numpy table-interp "
            "replica at rtol=1e-12"
        )
        rel = np.abs(got_t - exact) / exact
        print(
            f"RESULT esat_table_tiled backend={name} first={first_t:.2f}s "
            f"steady={steady_t * 1e3:.2f}ms max_rel_err_vs_analytic={rel.max():.2e}"
        )

        run_k_only_variant(name, backend, t=t, table=table_k_only, k_index=k_index, exact=exact)

    print("SPIKE D: PASS")


if __name__ == "__main__":
    main()

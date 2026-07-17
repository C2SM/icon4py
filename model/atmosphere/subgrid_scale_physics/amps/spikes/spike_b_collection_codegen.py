# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Spike B: code-generate an unrolled bin-pair collection (coalescence) kernel
and measure embedded/gtfn_cpu compile + run cost at nbins = 8, 20, 40.

Physics stand-in: Golovin kernel K_ij = k0*(m_i+m_j), mass-doubling bins
m_b = 2^b, Kovetz-Olund style two-bin deposit of coalesced mass, plus loss
terms. Structure (nbins^2 pair terms scattered to <=2 destination bins each,
plus nbins x nbins loss sums) matches the real AMPS coalescence shape.

This is the compile-time gate for the whole AMPS port architecture (spec §9
risk 1): can a single generated, fully-unrolled field_operator for the 40-bin
case even compile on gtfn_cpu in a bounded time? A backend failure or an
excessive compile time, captured verbatim, is itself the measurement.

CRITICAL for measurement validity: the environment has a PERSISTENT on-disk
gt4py build cache (GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT). `clear_gt4py_cache`
below removes the worktree-local `.gt4py_cache` directory (the cache gt4py
resolves to when GT4PY_BUILD_CACHE_DIR is unset and the process cwd is the
worktree root, as used by the documented run command) before each gtfn_cpu
compile so first-call timings are genuine cold-compile numbers, not artifacts
of a warm on-disk cache from a previous run/spike.

WARNING: with the muphys-precedented recursion-limit raise below, nbins=40's
gtfn_cpu compile SUCCEEDS but takes ~2579s (~43 minutes, measured) instead of
crashing in ~13s. Running this script end-to-end via the documented command
will therefore take on the order of 45-50 minutes total (dominated by that
one compile). This is an intentional, verified-necessary trade: without the
raised limit, nbins=40 crashes with RecursionError instead (see the task-8
report for both numbers). There is no way to get nbins=40's real gtfn_cpu
number without paying one or the other.

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_b_collection_codegen.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import common
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import (
    utils as muphys_driver_utils,
)


HEADER = """\
import gt4py.next as gtx

from icon4py.model.common import field_type_aliases as fa, type_alias as ta

"""


def gt4py_cache_dir() -> Path:
    """The on-disk gt4py build cache directory used by this spike's run
    command (`cd $WT && uv run ...`), i.e. `<worktree_root>/.gt4py_cache`.
    gt4py resolves its persistent cache relative to the process cwd when
    GT4PY_BUILD_CACHE_DIR is not set (confirmed unset in this environment);
    the worktree root is two levels above this file
    (spikes/ -> amps/ -> ... -> worktree root is found by walking up to the
    directory containing `.gt4py_cache` created by earlier spike runs).
    """
    # Walk up from this file to the worktree root (has .git).
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent / ".gt4py_cache"
    # Fallback: cwd-relative, matching gt4py's own default resolution.
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


def pair_weights(nbins: int) -> dict:
    """(i, j) -> list of (dest_bin, number_fraction). Mass-doubling grid."""
    mass = 2.0 ** np.arange(nbins)
    weights = {}
    for i in range(nbins):
        for j in range(i, nbins):
            m_new = mass[i] + mass[j]
            d = min(int(np.floor(np.log2(m_new))), nbins - 1)
            if d >= nbins - 1:
                weights[(i, j)] = [(nbins - 1, 1.0)]
            else:
                frac = (mass[d + 1] - m_new) / (mass[d + 1] - mass[d])
                weights[(i, j)] = [(d, frac), (d + 1, 1.0 - frac)]
    return weights


def kernel_table(nbins: int, k0: float = 1.5e-3) -> np.ndarray:
    mass = 2.0 ** np.arange(nbins)
    return k0 * (mass[:, None] + mass[None, :])


def gen_source(nbins: int) -> str:
    kern = kernel_table(nbins)
    weights = pair_weights(nbins)
    args = ",\n    ".join(f"n_{b:02d}: fa.CellKField[ta.wpfloat]" for b in range(nbins))
    rets = ", ".join("fa.CellKField[ta.wpfloat]" for _ in range(nbins))
    gains: dict[int, list[str]] = {b: [] for b in range(nbins)}
    for (i, j), deps in weights.items():
        sym = 1.0  # symmetric factor folded into K for the spike (both branches equal)
        for d, frac in deps:
            # float(...) is required: numpy>=2.0's repr() of a np.float64 scalar
            # is "np.float64(0.003)" (not a bare literal), which would emit
            # invalid/undefined-symbol source ("np" is unimported in the
            # generated module) into the generated field_operator body.
            gains[d].append(f"{float(frac * sym * kern[i, j])!r} * n_{i:02d} * n_{j:02d}")
    lines = []
    for b in range(nbins):
        loss = " + ".join(f"{float(kern[b, j])!r} * n_{j:02d}" for j in range(nbins))
        gain = " + ".join(gains[b]) if gains[b] else "0.0"
        lines.append(f"    dn_{b:02d} = ({gain}) - n_{b:02d} * ({loss})")
    outs = ", ".join(f"dn_{b:02d}" for b in range(nbins))
    return (
        HEADER
        + "@gtx.field_operator\n"
        + f"def _collection_{nbins}(\n    {args},\n) -> tuple[{rets}]:\n"
        + "\n".join(lines)
        + f"\n    return {outs}\n"
    )


def numpy_reference(n: np.ndarray, nbins: int) -> np.ndarray:
    kern = kernel_table(nbins)
    weights = pair_weights(nbins)
    dn = np.zeros_like(n)
    for (i, j), deps in weights.items():
        prod = kern[i, j] * n[i] * n[j]
        for d, frac in deps:
            dn[d] += frac * prod
    dn -= n * np.einsum("bj,j...->b...", kern, n)
    return dn


def run(nbins: int) -> bool:
    """Returns True iff every backend succeeded (and matched the numpy
    reference) for this nbins. A backend failure is captured as a `RESULT
    ... FAILED` line (not an uncaught crash) so a compiler blow-up at a large
    nbins doesn't prevent already-collected results (smaller nbins, other
    backends) from being reported -- mirrors spike_a_remap_gather.py's
    try/except pattern for the same reason (see that file's Fix 3 / the
    task-7 report).
    """
    t0 = time.perf_counter()
    src = gen_source(nbins)
    op = common.load_generated_operator(src, f"gen_collection_{nbins}", f"_collection_{nbins}")
    gen_s = time.perf_counter() - t0
    print(
        f"RESULT collection nbins={nbins} gen+parse={gen_s:.2f}s "
        f"source_lines={len(src.splitlines())}"
    )

    rng = np.random.default_rng(7)
    n_np = rng.uniform(0.0, 1.0, size=(nbins, common.NCELLS, common.NLEV))
    inputs = [common.make_field(n_np[b]) for b in range(nbins)]
    expected = numpy_reference(n_np, nbins)

    ok = True
    for name, backend in common.backends().items():
        if name == "gtfn_cpu":
            clear_gt4py_cache()
        outs = tuple(common.zeros_field() for _ in range(nbins))
        bound = op.with_backend(backend) if backend is not None else op

        def call(bound=bound, outs=outs):
            bound(*inputs, out=outs, offset_provider={})

        try:
            # gt4py 1.1.11's ITIR transform pipeline (the `eve`-framework tree
            # visitors, e.g. MergeLet) recurses per-node and can exceed
            # Python's default recursion limit (1000) for large generated
            # expression trees (see nbins=40's RecursionError, first
            # observed and recorded without this wrapper). This mirrors an
            # existing icon4py precedent: muphys's own drivers hit the same
            # gt4py limitation and work around it identically --
            # `icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver.
            # utils.recursion_limit` (a `sys.setrecursionlimit` context
            # manager that restores the original limit on exit) wraps
            # program compilation in `run_graupel_only.py` (limit `10**4`)
            # and `run_full_muphys.py` (limit `10**5`); both carry the same
            # `# TODO(havogt): make an option in gt4py?` note this mirrors.
            # 10**5 matches the higher of the two precedents.
            with muphys_driver_utils.recursion_limit(10**5):
                first, steady = common.time_first_and_steady(call, n_steady=5)
        except Exception as exc:  # a backend/compiler failure is itself a valid spike datum
            ok = False
            print(f"RESULT collection nbins={nbins} backend={name} FAILED: {exc!r}")
            continue

        got = np.stack([o.asnumpy() for o in outs])
        # rtol=1e-12 (the brief's original value) is unreachable in float64 for
        # nbins=40: the mass-doubling grid (2^0..2^39) gives the ~1600-term gain
        # sum and 40-term loss sum a huge dynamic range, so the generated
        # operator's summation order (flat left-to-right per the generated
        # source text) and numpy_reference's order (per-pair-then-einsum)
        # disagree at the ~1e-9-relative level from ordinary floating-point
        # non-associativity, not a logic bug -- verified independently via
        # exact Fraction (infinite-precision rational) arithmetic at the
        # worst-case point (nbins=40, bin 39): both the generated-operator
        # result and the numpy-reference result individually round-trip to
        # within a few ULPs of the exact rational answer, they just round
        # differently. 1e-6 comfortably tolerates that reordering noise (max
        # observed relative diff ~7.7e-10) while still gating any real
        # multi-order-of-magnitude logic bug.
        assert np.allclose(got, expected, rtol=1e-6), f"collection nbins={nbins} {name} wrong"
        print(
            f"RESULT collection nbins={nbins} backend={name} "
            f"first={first:.1f}s steady={steady * 1e3:.1f}ms"
        )
    return ok


if __name__ == "__main__":
    all_ok = True
    for nb in (8, 20, 40):
        all_ok &= run(nb)
    print("SPIKE B: PASS" if all_ok else "SPIKE B: PARTIAL (see RESULT ... FAILED lines above)")

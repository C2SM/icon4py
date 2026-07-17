# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""DSL-source-string builders for `codegen/generate.py::emit_module`.

Every builder here (and every future one, e.g. M2's collection kernel)
MUST emit SPLIT operators: one `@gtx.field_operator` per bin GROUP of at
most `chunk_size` (default 8) consecutive bins, never one monolithic
operator spanning all of a hydrometeor category's bins. See
`generate.py`'s module docstring for the full rationale (the M0 gate
report amendment; `spikes/spike_b_collection_codegen.py` measured a
single unrolled 40-bin operator's `gtfn_cpu` compile at ~2579s). This
module's own `chunk_bins()` is the shared chunking helper every builder
should use to satisfy that constraint, so builders don't each
re-implement the same slicing.

`build_axpy_per_bin` is M1's only builder: a trivial DEMONSTRATION
generator (`out_b = a_b * x_b + y_b` per bin `b`, `a_b` a baked-in float
literal) that exists purely to exercise `emit_module`/
`check_regenerated` end to end and to prove the chunking property
concretely (its own generated source has multiple field_operators once
`nbins > chunk_size`) -- it is NOT physics. The real chunked builder
(M2's collection kernel) replaces it as this package's first
non-demonstration codegen consumer.
"""

from __future__ import annotations

from collections.abc import Sequence


IMPORTS = "import gt4py.next as gtx\n\nfrom icon4py.model.common import field_type_aliases as fa, type_alias as ta\n"


def chunk_bins(nbins: int, chunk_size: int = 8) -> list[tuple[int, int]]:
    """Split `range(nbins)` into consecutive `[lo, hi)` groups of at
    most `chunk_size` bins each -- the SPLIT-operator chunking scheme
    this module's (and `generate.py`'s) docstring mandates for any
    codegen builder. `chunk_size` defaults to 8, the ceiling named in
    the M1 task brief/plan.
    """
    if nbins < 1:
        raise ValueError(f"nbins must be >= 1; got {nbins}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    return [(lo, min(lo + chunk_size, nbins)) for lo in range(0, nbins, chunk_size)]


def axpy_chunk_name(lo: int, hi: int) -> str:
    """Deterministic per-chunk field_operator name for
    `build_axpy_per_bin`'s output, shared with callers (e.g. tests) that
    need to look up a specific chunk's operator in the loaded module."""
    return f"_axpy_bins_{lo:02d}_{hi:02d}"


def default_axpy_coeffs(nbins: int) -> list[float]:
    """Deterministic per-bin AXPY coefficients -- a pure function of
    `nbins` only (no randomness), so `build_axpy_per_bin`'s output is
    reproducible byte-for-byte across re-emits, which
    `check_regenerated`'s drift guard depends on."""
    return [1.0 + 0.5 * b for b in range(nbins)]


def _build_axpy_chunk(lo: int, hi: int, coeffs: Sequence[float]) -> str:
    bins = list(range(lo, hi))
    arg_lines = []
    for b in bins:
        arg_lines.append(f"x_{b:02d}: fa.CellKField[ta.wpfloat],")
        arg_lines.append(f"y_{b:02d}: fa.CellKField[ta.wpfloat],")
    args_block = "\n    ".join(arg_lines)
    rets = ", ".join("fa.CellKField[ta.wpfloat]" for _ in bins)
    # float(...) is required: numpy>=2.0's repr() of a np.float64 scalar
    # is "np.float64(1.5)" (not a bare literal), which would emit
    # invalid/undefined-symbol source ("np" is unimported in the
    # generated module) -- see spike_b_collection_codegen.py's identical
    # comment, the precedent this line follows.
    body_lines = [f"    out_{b:02d} = {float(coeffs[b])!r} * x_{b:02d} + y_{b:02d}" for b in bins]
    outs = ", ".join(f"out_{b:02d}" for b in bins)
    # The generated `def` line below carries a "noqa: PLR0917" marker
    # ("too many positional arguments") -- a chunk's field params (2 per
    # bin, up to 16 at chunk_size=8) are exactly the SPLIT-operator design
    # this module targets, not something to shrink; spike_e_counter_rng.py's
    # `_hash01` carries the identical suppression for the same reason
    # (module-level scalar params, not a smell).
    return (
        "@gtx.field_operator\n"
        f"def {axpy_chunk_name(lo, hi)}(  # noqa: PLR0917\n    {args_block}\n) -> tuple[{rets}]:\n"
        + "\n".join(body_lines)
        + f"\n    return {outs}\n"
    )


def build_axpy_per_bin(
    nbins: int, *, chunk_size: int = 8, coeffs: Sequence[float] | None = None
) -> str:
    """M1's trivial DEMONSTRATION codegen builder: `y_b = a_b*x_b + y_b`
    per bin `b`, one `@gtx.field_operator` PER CHUNK of at most
    `chunk_size` (default 8) consecutive bins -- SPLIT operators, never
    one monolithic N-bin operator (see module docstring). Proves the
    `emit_module()`/`check_regenerated()` pipeline end-to-end; it is NOT
    physics.

    Args:
        nbins: total number of bins.
        chunk_size: max bins per generated field_operator (default 8).
        coeffs: per-bin AXPY coefficients; defaults to
            `default_axpy_coeffs(nbins)` (deterministic, no randomness --
            required for reproducible re-emits).
    """
    if coeffs is None:
        coeffs = default_axpy_coeffs(nbins)
    if len(coeffs) != nbins:
        raise ValueError(f"len(coeffs)={len(coeffs)} != nbins={nbins}")

    blocks = [IMPORTS.rstrip("\n")]
    for lo, hi in chunk_bins(nbins, chunk_size):
        blocks.append(_build_axpy_chunk(lo, hi, coeffs).rstrip("\n"))
    return "\n\n\n".join(blocks) + "\n"

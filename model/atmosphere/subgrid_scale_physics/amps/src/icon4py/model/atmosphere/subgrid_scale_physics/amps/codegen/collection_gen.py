# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""DSL-source-string builder for the collection (coalescence) RATE-MATRIX
kernel -- M2b Task 4, the first REAL (non-demonstration) consumer of
`codegen/generate.py`/`codegen/templates.py` (M1 Task 6), per the module
docstrings of both.

DSL / numpy split (read this first -- the central decision of this
module):

`core/coalescence.py::coalesce_rain` (M2b Task 3, the reference numpy
engine) is NOT codegen'd here, and cannot usefully be: its
`collector_loop1` (H/F/LM sub-population categorization, the
`used_marker` fixed point across up to `nbins` rounds, the multi-bin PDF
scatter) is genuinely data-dependent control flow -- per-pair branching
and an iterative fixed point with a data-dependent trip count -- none of
which gt4py's declarative field_operator DSL expresses (no data-dependent
`while`, no per-point branching beyond `where()`). That entire engine
stays numpy, unchanged, SANCTIONED per the M0 gate report and this task's
own dispatch.

What IS DSL-expressible, and what this module codegens, is the inner
per-bin-pair RATE assembly `core/collision_kernel.py::collision_kernel`
performs before any of that scatter/categorization happens:

    KC_ij = E_c_ij * (vtm_i - vtm_j) * A_c_ij * con_j * dt
    A_c_ij = 0.25 * pi * (len_i + len_j)**2   -- col_level==1 branch only

This is a pure, embarrassingly-parallel elementwise expression over the
`(nbins_i, nbins_j, ncells, nlev)` bin-pair rate matrix: no data-dependent
branching, no iteration, no gather -- exactly the "pointwise sub-computation"
this task's dispatch scopes as DSL-expressible, and the concrete O(nbins^2)
shape `spike_b_collection_codegen.py` (F5/M0) measured compile cost for.
`col_level==0` (the alternate `A_c` branch, `A_c=0` whenever either length
exceeds 1 micron) is deliberately NOT generated: cloudlab's own config
(`AmpsConfig.cloudlab().coll_level == 1`) never exercises it, matching
`collision_kernel.py`'s own docstring note; nothing here technically
prevents adding it (it's one more `where()`), it is just out of this
task's realistic scope.

`E_c_ij` (collision efficiency) is deliberately NOT computed inside the
generated operators -- it is threaded in as a plain per-pair INPUT field,
still produced by the existing numpy `collision_efficiency`
(`core/collision_kernel.py`, unchanged). Two independent, concrete reasons,
not just caution:

1. `collision_efficiency` is a genuine 2D bilinear-interpolation gather
   against `AmpsLuts.drpdrp` (measured shape `(201, 201)` = 40401 entries)
   at TWO independently-varying, per-bin-pair, per-point computed indices
   (row from `log10(Nre)`, column from the length ratio) -- a 4-corner
   stencil lookup, not a single-index gather.
2. `spikes/spike_a_remap_gather.py` (M0/spike-A) is the only precedent for
   ANY table gather in this DSL, and its own finding is narrower than that:
   a plain K-only table gather is a NO-GO on both embedded and gtfn_cpu
   (decoration-time `DSLError`, dims mismatch); the only validated
   workaround is `_table_gather_tiled` -- tiling the table to full
   `(Cell, K)` shape and gathering via a SINGLE computed index along a
   Cartesian `Koff` self-offset. That idiom was only ever exercised for a
   K-sized (~61-entry) 1D table with ONE index. Extending it to `drpdrp`
   would need (a) a second, independent gather axis (nothing in gt4py
   1.1.11's `Koff`/`as_offset` machinery gathers two axes at once -- it
   would require flattening the table to 1D and reconstructing a combined
   index, an unvalidated extension), and (b) tiling 40401 entries to full
   `(Cell, K)` shape, i.e. `NCELLS * 40401 * 8` bytes per table copy
   (~1.3 GB at the spike's own `NCELLS=4096`) for JUST ONE of `AmpsLuts`'s
   seven collision-efficiency tables. Neither is a scale-preserving
   extension of what spike A actually validated. Keeping the gather numpy
   (computed once, up front, exactly as `coalesce_rain` already does) is
   the correct engineering call, not a shortcut -- see
   `.superpowers/sdd/m2b-task-4-report.md` for the full writeup.

Chunking: SPLIT operators per the M1 codegen skeleton's binding
constraint (`generate.py`'s module docstring) -- at most `chunk_size`
(default 8, `templates.chunk_bins`'s own default) bins per operator. The
rate matrix has TWO bin axes (collector `i`, collectee `j`), so this
module chunks BOTH via `templates.chunk_bins`, giving one field_operator
per `(i_chunk, j_chunk)` PAIR -- at most `chunk_size**2` (64 at the
default) bin-PAIRS per operator, the direct 2D generalization of the 1D
per-bin chunking `build_axpy_per_bin` already established. At `nbins=40`
(cloudlab's `AmpsConfig.cloudlab().num_h_bins[0]`, the M0-gate size) with
the default `chunk_size=8` this is `(40/8)**2 = 25` operators, each with
`8*2 + 8*3 + 8*8 = 104` field params (`len_i`/`vtm_i` for the i-chunk's <=8
bins, `len_j`/`vtm_j`/`con_j` for the j-chunk's <=8 bins, `ec_i.._j..` for
every pair in the chunk) plus a scalar `dt` -- see
`.superpowers/sdd/m2b-task-4-report.md` for the measured gtfn_cpu compile
numbers at this size (this environment DOES have a working C++/ninja/cmake
toolchain, so gtfn feasibility was actually measured, not just
toolchain-gated).
"""

from __future__ import annotations

from icon4py.model.atmosphere.subgrid_scale_physics.amps.codegen.templates import (
    IMPORTS,
    chunk_bins,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst


# H1 SS2 (`mod_amps_core.F90:15913-15942`), col_level==1 branch:
# A_c = 0.25*pi*(len_i+len_j)**2, unconditional. Baked as a compile-time
# constant (PI never varies) -- see module docstring for why col_level==0
# is out of this builder's scope.
_A_C_COEF = 0.25 * float(AmpsConst.PI)


def kc_chunk_name(ilo: int, ihi: int, jlo: int, jhi: int) -> str:
    """Deterministic per-(i_chunk,j_chunk) field_operator name, shared
    with callers (e.g. tests) that need to look up a specific chunk's
    operator in the loaded module."""
    return f"_kc_bins_i{ilo:02d}_{ihi:02d}_j{jlo:02d}_{jhi:02d}"


def _build_kc_chunk(ilo: int, ihi: int, jlo: int, jhi: int) -> str:
    i_bins = list(range(ilo, ihi))
    j_bins = list(range(jlo, jhi))

    arg_lines = []
    for i in i_bins:
        arg_lines.append(f"len_i{i:02d}: fa.CellKField[ta.wpfloat],")
        arg_lines.append(f"vtm_i{i:02d}: fa.CellKField[ta.wpfloat],")
    for j in j_bins:
        arg_lines.append(f"len_j{j:02d}: fa.CellKField[ta.wpfloat],")
        arg_lines.append(f"vtm_j{j:02d}: fa.CellKField[ta.wpfloat],")
        arg_lines.append(f"con_j{j:02d}: fa.CellKField[ta.wpfloat],")
    arg_lines.extend(
        f"ec_i{i:02d}_j{j:02d}: fa.CellKField[ta.wpfloat]," for i in i_bins for j in j_bins
    )
    # dt: a genuine runtime scalar (the collision-substep timestep), NOT
    # baked as a literal -- `ta.wpfloat` scalar field_operator params are
    # an established gt4py pattern (see e.g. muphys's
    # core/properties.py::vel_scale_factor_default, `dt: ta.wpfloat`).
    arg_lines.append("dt: ta.wpfloat,")
    args_block = "\n    ".join(arg_lines)

    outs: list[str] = []
    body_lines: list[str] = []
    for i in i_bins:
        for j in j_bins:
            out_name = f"kc_i{i:02d}_j{j:02d}"
            outs.append(out_name)
            # float(...) is required: numpy>=2.0's repr() of a np.float64
            # scalar is "np.float64(0.785...)" (not a bare literal), which
            # would emit invalid/undefined-symbol source ("np" is
            # unimported here) -- same precedent as
            # templates.py::_build_axpy_chunk and
            # spike_b_collection_codegen.py::gen_source.
            a_c = f"({float(_A_C_COEF)!r} * (len_i{i:02d} + len_j{j:02d}) ** 2.0)"
            body_lines.append(
                f"    {out_name} = ec_i{i:02d}_j{j:02d} * (vtm_i{i:02d} - vtm_j{j:02d}) "
                f"* {a_c} * con_j{j:02d} * dt"
            )

    rets = ", ".join("fa.CellKField[ta.wpfloat]" for _ in outs)
    outs_str = ", ".join(outs)
    # The generated def line below always suppresses the "too-many-
    # positional-arguments" lint code (PLR0917): the many per-bin-pair
    # field params ARE the intended SPLIT-operator design (see module
    # docstring), same precedent as templates.py's axpy chunks. Once
    # chunk_size**2 exceeds the repo's max-statements=60 (chunk_size=8 ->
    # 64 body assignments + 1 return = 65 statements), the "too-many-
    # statements" lint code is ALSO suppressed -- same precedent as
    # core/coalescence.py::coalesce_rain's own multi-code suppression
    # marker on its def line; here it's "one assignment per bin-pair", the
    # split-operator analogue.
    codes = "PLR0915, PLR0917" if len(outs) > 60 else "PLR0917"
    noqa = f"  # noqa: {codes}"
    return (
        "@gtx.field_operator\n"
        f"def {kc_chunk_name(ilo, ihi, jlo, jhi)}({noqa}\n    {args_block}\n"
        f") -> tuple[{rets}]:\n" + "\n".join(body_lines) + f"\n    return {outs_str}\n"
    )


def build_collision_rate_matrix(nbins: int, *, chunk_size: int = 8) -> str:
    """M2b Task 4's real (non-demonstration) codegen builder: the
    collector-collectee RATE MATRIX `KC_ij = E_c_ij*(vtm_i-vtm_j)*
    A_c_ij*con_j*dt` (col_level==1 only), one `@gtx.field_operator` PER
    `(i_chunk, j_chunk)` PAIR of at most `chunk_size` (default 8)
    consecutive bins each -- SPLIT operators on BOTH bin axes (see module
    docstring). `E_c_ij` is a plain input field per pair, NOT computed
    here (see module docstring's DSL/numpy split).

    Args:
        nbins: total number of liquid bins (cloudlab: 40).
        chunk_size: max bins per axis per generated field_operator
            (default 8, `templates.chunk_bins`'s own default).
    """
    blocks = [IMPORTS.rstrip("\n")]
    i_chunks = chunk_bins(nbins, chunk_size)
    j_chunks = chunk_bins(nbins, chunk_size)
    for ilo, ihi in i_chunks:
        for jlo, jhi in j_chunks:
            blocks.append(_build_kc_chunk(ilo, ihi, jlo, jhi).rstrip("\n"))
    return "\n\n\n".join(blocks) + "\n"

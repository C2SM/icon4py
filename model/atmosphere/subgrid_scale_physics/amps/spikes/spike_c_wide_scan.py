# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Spike C: scan_operator with a 40-field NamedTuple carry — the shape of
40-bin implicit sedimentation (muphys does 4 hydrometeor classes; AMPS needs
40 bins). Generated source (explicit 40-field NamedTuple + 40 scalar args).

Numerics stand-in (per level, forward/downward scan, muphys-style):
  flux_b_new = v_b * q_b * rho ; q_out_b = q_b + zeta*(carry.f_b - flux_b_new)
carry = fluxes. Correctness checked vs numpy recurrence.

CRITICAL for measurement validity: the environment has a PERSISTENT on-disk
gt4py build cache (GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT). `clear_gt4py_cache`
below removes the worktree-local `.gt4py_cache` directory (the cache gt4py
resolves to when GT4PY_BUILD_CACHE_DIR is unset and the process cwd is the
worktree root, as used by the documented run command) before the gtfn_cpu
compile so the first-call timing is a genuine cold-compile number, not an
artifact of a warm on-disk cache from a previous run/spike (same idiom as
spike_b_collection_codegen.py; the helper functions are copied verbatim
below since spikes are standalone scripts).

SCAN OUTPUT CONVENTION (measured, not assumed): gt4py 1.1.11's scan_operator
emits the *post-update* carry at each level -- the NamedTuple a scan step
*returns* is both next level's input carry AND the value written to this
level's output field. Verified by direct comparison against both a
post-update and a pre-update (one-level-shifted, output[0]=init) numpy
recurrence at NBINS in {3, 40}, NCELLS in {2, 5, 16, 64, 256}: got matches
the post-update recurrence exactly (max abs diff 0.0, not just within
rtol=1e-12) and mismatches the pre-update one by O(1) (not rounding noise).
numpy_reference below already implements the post-update recurrence, so no
flip was needed versus the code as drafted in the task brief.

MEASURED (this spike's actual go/no-go datum): gtfn_cpu -- clear-cache cold
compile succeeds in 82.7s, steady-state run 93.8ms/call (NBINS=40,
NCELLS=4096, NLEV=61; recursion_limit(10**5) applied per the muphys driver
precedent, though unlike spike_b's O(nbins^2) collection kernel this scan's
per-level body is O(nbins) -- one flux + one (unused-downstream, see
gen_source) update expression per bin -- so it never approached the
recursion-limit-adjacent slowdown spike_b hit at the same nbins; a clean GO
for gtfn_cpu at this width).

embedded, by contrast, is NOT a clean measurement at this spike's full scale:
a run at the documented NCELLS=4096/NLEV=61/NBINS=40 size ran for ~95 minutes
at ~100% CPU and then the process was simply gone -- no RESULT line, no
Python traceback (stderr was merged into the same captured stream), nothing.
Not confirmed via OS logs, but consistent with an out-of-memory kill (macOS
jetsam delivers SIGKILL, which allows no Python-level cleanup or traceback).
Smaller-scale embedded calibration (NCELLS in {16, 64, 256}, same
NBINS=40/NLEV=61) DID complete and confirmed exact numerical correctness
(post-update convention, max abs diff 0.0) with an apparently linear
~55-56 ms/cell single-call cost -- naive linear extrapolation predicts
~225s/call at NCELLS=4096, which does not remotely reconcile with the
observed ~95-minute non-completion, so whatever embedded's true scaling is
at production width x production grid size, it is worse than linear (or
some other size-dependent effect -- e.g. memory pressure from materializing
40 NamedTuple-typed (Cell, K) intermediate fields at once -- dominates)
Given gtfn_cpu's clean, fast result above already answers this spike's
actual question (compile/run cost of the *compiled* path, which is what any
real driver would use), no further attempt was made to force a full-scale
embedded completion; see the task-9 report for the full timeline.

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_c_wide_scan.py
(add AMPS_SPIKE_C_GTFN_ONLY=1 to measure gtfn_cpu alone, skipping the
impractical full-scale embedded pass documented above; AMPS_SPIKE_C_EMBEDDED_ONLY=1
for the reverse.)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import common
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import (
    utils as muphys_driver_utils,
)


NBINS = 40
VTS = np.linspace(0.1, 8.0, NBINS)  # per-bin fall speeds, folded as constants


def gt4py_cache_dir() -> Path:
    """The on-disk gt4py build cache directory used by this spike's run
    command (`cd $WT && uv run ...`), i.e. `<worktree_root>/.gt4py_cache`.
    Copied from spike_b_collection_codegen.py (spikes are standalone scripts).
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


def gen_source(nbins: int) -> str:
    header = (
        "from typing import NamedTuple\n\n"
        "import gt4py.next as gtx\n\n"
        "from icon4py.model.common import dimension as dims, type_alias as ta\n\n\n"
    )
    carry_fields = "\n".join(f"    f_{b:02d}: ta.wpfloat" for b in range(nbins))
    carry = f"class Carry{nbins}(NamedTuple):\n{carry_fields}\n\n\n"
    init = ", ".join(f"f_{b:02d}=0.0" for b in range(nbins))
    qargs = ",\n    ".join(f"q_{b:02d}: ta.wpfloat" for b in range(nbins))
    # float(...) is required: numpy>=2.0's repr() of a np.float64 scalar is
    # "np.float64(0.1)" (not a bare literal), which would emit invalid/
    # undefined-symbol source ("np" is unimported in the generated module)
    # into the generated scan_operator body. Same fix as
    # spike_b_collection_codegen.py's gen_source (VTS[b] here is a np.float64
    # element of a np.linspace array, same failure mode as that spike's
    # kernel-table entries).
    body_flux = "\n".join(
        f"    fl_{b:02d} = {float(VTS[b])!r} * q_{b:02d} * rho" for b in range(nbins)
    )
    body_q = "\n".join(
        f"    qo_{b:02d} = q_{b:02d} + zeta * (carry.f_{b:02d} - fl_{b:02d})" for b in range(nbins)
    )
    ret_carry = ", ".join(f"f_{b:02d}=fl_{b:02d}" for b in range(nbins))
    outs_tuple = ", ".join(f"s.f_{b:02d}" for b in range(nbins))
    rets = ", ".join(
        "gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat]" for _ in range(nbins)
    )
    fargs = ",\n    ".join(
        f"q_{b:02d}: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat]" for b in range(nbins)
    )
    fcall = ", ".join(f"q_{b:02d}" for b in range(nbins))
    return (
        header
        + carry
        + f"@gtx.scan_operator(axis=dims.KDim, forward=True, init=Carry{nbins}({init}))\n"
        + f"def _sed_scan_{nbins}(\n    carry: Carry{nbins},\n    {qargs},\n"
        + "    rho: ta.wpfloat,\n    zeta: ta.wpfloat,\n"
        + f") -> Carry{nbins}:\n"
        + body_flux
        + "\n"
        + body_q
        + "\n"
        + f"    return Carry{nbins}({ret_carry})\n\n\n"
        + "@gtx.field_operator\n"
        + f"def _sed_{nbins}(\n    {fargs},\n"
        + "    rho: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],\n"
        + "    zeta: ta.wpfloat,\n"
        + f") -> tuple[{rets}]:\n"
        + f"    s = _sed_scan_{nbins}({fcall}, rho, zeta)\n"
        + f"    return {outs_tuple}\n"
    )


def numpy_reference(q: np.ndarray, rho: np.ndarray, zeta: float) -> np.ndarray:
    # Confirmed against the generated scan_operator (see the module docstring
    # "scan output convention" note): gt4py emits the *post-update* carry
    # (the value returned by the scan body at level k) as level k's output,
    # matching this recurrence exactly -- max abs diff 0.0 at small-scale
    # embedded calibration (NBINS in {3, 40}, NCELLS in {2, 5, 16, 64, 256},
    # NLEV in {4, 61}) and rtol=1e-12 at the full documented scale (NBINS=40,
    # NCELLS=4096, NLEV=61) via the gtfn_cpu backend's assert in main() (the
    # embedded backend did not complete at that full scale -- see the module
    # docstring) -- no flip needed. fl at level k does not actually depend on
    # the carry passed into that level (only the unused qo_b intermediate in
    # the generated body does), so there is no `carry` variable to track
    # here; ruff flagged the brief's original `carry = np.zeros(...)` /
    # `carry = fl` bookkeeping as unused (F841) once written this way, which
    # is correct -- it really is unused.
    out = np.empty_like(q)
    for k in range(q.shape[2]):
        out[:, :, k] = VTS[:, None] * q[:, :, k] * rho[None, :, k]
    return out


def active_backends() -> dict:
    """common.backends(), optionally filtered to embedded-only when
    AMPS_SPIKE_C_EMBEDDED_ONLY is set, or gtfn_cpu-only when
    AMPS_SPIKE_C_GTFN_ONLY is set (either as any non-empty value). Same
    escape hatch as spike_b_collection_codegen.py's active_backends: lets one
    backend be re-run in isolation without paying the other's cost. Not part
    of the brief; added for iterative development only -- the committed
    measurement run uses the default (both backends). The GTFN_ONLY variant
    was added mid-spike after the full-scale (NCELLS=4096, NLEV=61) embedded
    run turned out to be impractically expensive (see the report/module
    docstring for the observed numbers) -- it lets the (much more important,
    for the compile-time go/no-go question) gtfn_cpu measurement be obtained
    without first paying the embedded backend's cost in the same process.
    """
    backends = common.backends()
    if os.environ.get("AMPS_SPIKE_C_EMBEDDED_ONLY"):
        backends = {"embedded": backends["embedded"]}
    elif os.environ.get("AMPS_SPIKE_C_GTFN_ONLY"):
        backends = {"gtfn_cpu": backends["gtfn_cpu"]}
    return backends


def main() -> None:
    src = gen_source(NBINS)
    op = common.load_generated_operator(src, f"gen_sed_{NBINS}", f"_sed_{NBINS}")
    rng = np.random.default_rng(11)
    q_np = rng.uniform(0.0, 1e-3, size=(NBINS, common.NCELLS, common.NLEV))
    rho_np = rng.uniform(0.8, 1.2, size=(common.NCELLS, common.NLEV))
    inputs = [common.make_field(q_np[b]) for b in range(NBINS)]
    rho = common.make_field(rho_np)
    expected = numpy_reference(q_np, rho_np, 0.5)

    ok = True
    for name, backend in active_backends().items():
        if name == "gtfn_cpu":
            clear_gt4py_cache()
        outs = tuple(common.zeros_field() for _ in range(NBINS))
        bound = op.with_backend(backend) if backend is not None else op

        def call(bound=bound, outs=outs):
            bound(*inputs, rho, 0.5, out=outs, offset_provider={})

        try:
            # See spike_b_collection_codegen.py's run() for the full
            # rationale (muphys driver precedent): gt4py 1.1.11's ITIR
            # transform pipeline recurses per-node and can exceed Python's
            # default recursion limit for large generated expression trees.
            with muphys_driver_utils.recursion_limit(10**5):
                first, steady = common.time_first_and_steady(call, n_steady=5)
        except Exception as exc:  # a backend/compiler failure is itself a valid spike datum
            ok = False
            print(f"RESULT wide_scan nbins={NBINS} backend={name} FAILED: {exc!r}")
            continue

        got = np.stack([o.asnumpy() for o in outs])
        assert np.allclose(got, expected, rtol=1e-12), f"wide scan {name} wrong"
        print(
            f"RESULT wide_scan nbins={NBINS} backend={name} "
            f"first={first:.1f}s steady={steady * 1e3:.1f}ms"
        )
    print("SPIKE C: PASS" if ok else "SPIKE C: PARTIAL (see RESULT ... FAILED lines above)")


if __name__ == "__main__":
    main()

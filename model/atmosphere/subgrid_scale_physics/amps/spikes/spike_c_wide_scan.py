# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Spike C: scan_operator with an 80-field NamedTuple carry (2x40, see "FIX"
below) -- the shape of 40-bin implicit sedimentation (muphys does 4
hydrometeor classes; AMPS needs 40 bins). Generated source (explicit
80-field NamedTuple + 40 scalar args).

Numerics stand-in (per level, forward/downward scan, muphys-style):
  flux_b_new = v_b * q_b * rho ; q_out_b = q_b + zeta*(carry.f_b - flux_b_new)
carry = (fluxes, updated q). Correctness checked vs numpy recurrence.
Perturbation check (run_perturbation_check, always run): a level-0
perturbation must propagate into level 1's output, proving this is a
genuine sequential scan and not a degenerate level-parallel computation.

FIX (post-review): the first version of this spike (commit c2702ec60) had a
degenerate carry -- the exposed output (`fl_b`) never read `carry` at all
(only the *unused* `qo_b` intermediate did), so the whole "scan" was
mathematically 40 independent per-level pointwise computations with no
actual cross-level data dependency, and its 82.7s gtfn_cpu compile number
measured that degenerate case, not a genuine wide-carry scan. Fixed by:
widening the carry to 80 fields (f_b: flux, q_b: updated q -- q_b is
carried but deliberately not read back, per review: this stress-tests
carry *width* without adding a second live recurrence) and exposing
`s.q_XX` (which genuinely reads `carry.f_b`) as the field_operator's output
instead of `s.f_XX`. See "MEASURED" below for the corrected number.

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
level's output field. numpy_reference below implements this convention
(prev_flux carried in, this level's fl computed and stashed for next level,
this level's *output* is the carry-dependent qo). See run_perturbation_check
for the executable proof this is genuinely sequential.

MEASURED (this spike's actual go/no-go datum, corrected/superseded number --
see FIX above; the original 82.7s measured a degenerate, non-sequential
computation): gtfn_cpu -- clear-cache cold compile succeeds in 186.9s,
steady-state run 115.9ms/call (NBINS=40 i.e. an 80-field carry, NCELLS=4096,
NLEV=61; recursion_limit(10**5) applied per the muphys driver precedent).
Still a clean GO by spike_b_collection_codegen.py's thresholds (<120s clean
go, 120s-15min go-with-caching, >15min/crash no-go) -- 186.9s lands just
past the "clean" boundary into "go with per-kernel compile caching" (which
run_gtfn_cached already provides within a process), and is ~2.3x the
degenerate 82.7s number, consistent with the genuinely sequential carry
(now threading `carry.f_b` into the exposed output, plus double the carry
field count) costing more to compile/lower than the old level-parallel
computation, while still being ~14x faster than spike_b's O(nbins^2)
collection kernel at the same nbins=40 (2578.7s).

embedded is NOT attempted at full scale by default: a prior full-scale
(NCELLS=4096/NLEV=61) run of this spike's (pre-fix) generated operator ran
for ~90-95 minutes at ~100% CPU and then the process was simply gone -- no
RESULT line, no Python traceback (stderr was merged into the same captured
stream), nothing; not confirmed via OS logs, but consistent with an
out-of-memory kill. See the task-9 report for the full timeline. Given that
observed cost/risk, embedded now defaults to a small-NCELLS calibration
ladder (reproducible from this committed script, see EMBEDDED_CALIBRATION_NCELLS)
instead of the full documented grid size; set AMPS_SPIKE_C_FULL_EMBEDDED=1 to
attempt the full common.NCELLS scale anyway (not recommended without a
process-level time/memory guard).

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_c_wide_scan.py
(gtfn_cpu now runs first, by default, followed by the small-NCELLS embedded
calibration ladder. AMPS_SPIKE_C_GTFN_ONLY=1 / AMPS_SPIKE_C_EMBEDDED_ONLY=1
isolate one backend; AMPS_SPIKE_C_FULL_EMBEDDED=1 opts into the
(discouraged, see above) full-scale embedded attempt.)
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

# embedded default: a small-NCELLS calibration ladder (see module docstring
# for why full-scale embedded is opt-in, not default). Chosen to match the
# range independently calibrated during the original (pre-fix) spike run
# (linear ~55-56 ms/cell at these sizes for the degenerate carry; re-checked
# below for the fixed, genuinely-sequential carry).
EMBEDDED_CALIBRATION_NCELLS = (16, 64, 256)


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
    """Generate a scan_operator with a genuinely sequential carry: the
    exposed output `qo_b` reads `carry.f_b` (the previous level's flux), so
    each level's output depends on the previous level's input -- unlike the
    pre-fix version, which exposed `fl_b` (never reads `carry`) and left the
    carry-consuming `qo_b` as dead code. The carry additionally stores
    `q_b` (=qo_b) per review request, widening it to 2*nbins fields as a
    stress test of carry *width* -- `carry.q_b` is deliberately never read
    back (there is no second live recurrence), only stored.
    """
    header = (
        "from typing import NamedTuple\n\n"
        "import gt4py.next as gtx\n\n"
        "from icon4py.model.common import dimension as dims, type_alias as ta\n\n\n"
    )
    carry_fields = "\n".join(
        f"    f_{b:02d}: ta.wpfloat\n    q_{b:02d}: ta.wpfloat" for b in range(nbins)
    )
    carry = f"class Carry{nbins}(NamedTuple):\n{carry_fields}\n\n\n"
    init = ", ".join(f"f_{b:02d}=0.0, q_{b:02d}=0.0" for b in range(nbins))
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
    # qo_b genuinely reads carry.f_b (previous level's flux) -- this is the
    # value exposed as the field_operator's output below, so the scan is
    # actually sequential (see run_perturbation_check for the proof).
    body_q = "\n".join(
        f"    qo_{b:02d} = q_{b:02d} + zeta * (carry.f_{b:02d} - fl_{b:02d})" for b in range(nbins)
    )
    ret_carry = ", ".join(f"f_{b:02d}=fl_{b:02d}, q_{b:02d}=qo_{b:02d}" for b in range(nbins))
    outs_tuple = ", ".join(f"s.q_{b:02d}" for b in range(nbins))
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
    """Matches gen_source's genuinely-sequential recurrence: level k's output
    is `q_k + zeta * (prev_flux - fl_k)`, where `prev_flux` is the flux
    computed at level k-1 (0 at k=0, matching Carry's init=0.0). This is the
    carry-dependent `qo_b` that gen_source now exposes as `s.q_XX` -- the
    pre-fix version of this function computed only `fl` (independent of any
    carried state), matching the pre-fix generator's (degenerate) `s.f_XX`
    output; see the module docstring's FIX note.
    """
    out = np.empty_like(q)
    prev_flux = np.zeros((q.shape[0], q.shape[1]))
    for k in range(q.shape[2]):
        fl_k = VTS[:, None] * q[:, :, k] * rho[None, :, k]
        out[:, :, k] = q[:, :, k] + zeta * (prev_flux - fl_k)
        prev_flux = fl_k
    return out


def run_perturbation_check() -> None:
    """Reviewer's probe method, committed as an executable check: at a tiny
    scale (NBINS=3, NCELLS=2, NLEV=4, embedded -- fast), perturb q at level 0
    only and verify the perturbation propagates into level 1's *output*.
    Level 1's own q input is untouched by the perturbation, so if level 1's
    output changes, that can only be because the scan actually threaded
    `carry.f_b` from level 0 into level 1's `qo_b` computation -- i.e. this
    is a genuine sequential scan, not the degenerate level-parallel
    computation the pre-fix version of this spike measured (where this
    exact assertion would have failed: level 1 would have been unaffected,
    since the pre-fix `fl_b` output never read `carry` at all).
    """
    nb, nc, nl = 3, 2, 4
    src = gen_source(nb)
    op = common.load_generated_operator(src, f"gen_sed_perturb_{nb}", f"_sed_{nb}")
    rng = np.random.default_rng(99)
    q_np = rng.uniform(0.0, 1e-3, size=(nb, nc, nl))
    rho_np = rng.uniform(0.8, 1.2, size=(nc, nl))
    zeta = 0.5

    def run(q_arr: np.ndarray) -> np.ndarray:
        inputs = [common.make_field(q_arr[b]) for b in range(nb)]
        rho = common.make_field(rho_np)
        outs = tuple(common.zeros_field(shape=(nc, nl)) for _ in range(nb))
        op(*inputs, rho, zeta, out=outs, offset_provider={})
        return np.stack([o.asnumpy() for o in outs])

    baseline = run(q_np)
    perturbed_q = q_np.copy()
    perturbed_q[:, :, 0] += 1.0  # large, level-0-only perturbation
    perturbed = run(perturbed_q)

    assert not np.allclose(perturbed[:, :, 0], baseline[:, :, 0]), (
        "perturbation check: level 0 output did not change -- test itself is broken"
    )
    assert not np.allclose(perturbed[:, :, 1], baseline[:, :, 1]), (
        "perturbation at level 0 did not propagate to level 1 -- carry is dead "
        "(this is exactly the degenerate-carry bug the FIX addresses)"
    )
    print(
        "PERTURBATION CHECK: level-0 perturbation propagates into level 1 via "
        "carry (genuine sequential scan) -- PASS"
    )


def active_backends() -> dict:
    """common.backends(), optionally filtered to embedded-only when
    AMPS_SPIKE_C_EMBEDDED_ONLY is set, or gtfn_cpu-only when
    AMPS_SPIKE_C_GTFN_ONLY is set (either as any non-empty value). Same
    escape hatch as spike_b_collection_codegen.py's active_backends: lets one
    backend be re-run in isolation without paying the other's cost. Not part
    of the brief; added mid-spike, once full-scale embedded turned out to be
    impractical, to let the gtfn_cpu measurement be obtained without paying
    embedded's cost in the same process. The committed default (both flags
    unset) now runs gtfn_cpu first, then the embedded calibration ladder --
    see main().
    """
    backends = common.backends()
    if os.environ.get("AMPS_SPIKE_C_EMBEDDED_ONLY"):
        return {"embedded": backends["embedded"]}
    if os.environ.get("AMPS_SPIKE_C_GTFN_ONLY"):
        return {"gtfn_cpu": backends["gtfn_cpu"]}
    return backends


def measure_gtfn(op) -> bool:
    """Full documented scale (common.NCELLS x common.NLEV) gtfn_cpu
    measurement -- this spike's primary datum. Always attempted first when
    active (safe/bounded, unlike full-scale embedded -- see module
    docstring). Returns True iff it succeeded and matched numpy_reference.
    """
    clear_gt4py_cache()
    rng = np.random.default_rng(11)
    q_np = rng.uniform(0.0, 1e-3, size=(NBINS, common.NCELLS, common.NLEV))
    rho_np = rng.uniform(0.8, 1.2, size=(common.NCELLS, common.NLEV))
    inputs = [common.make_field(q_np[b]) for b in range(NBINS)]
    rho = common.make_field(rho_np)
    expected = numpy_reference(q_np, rho_np, 0.5)
    outs = tuple(common.zeros_field() for _ in range(NBINS))
    bound = op.with_backend(common.backends()["gtfn_cpu"])

    def call() -> None:
        bound(*inputs, rho, 0.5, out=outs, offset_provider={})

    try:
        # See spike_b_collection_codegen.py's run() for the full rationale
        # (muphys driver precedent): gt4py 1.1.11's ITIR transform pipeline
        # recurses per-node and can exceed Python's default recursion limit
        # for large generated expression trees.
        with muphys_driver_utils.recursion_limit(10**5):
            first, steady = common.time_first_and_steady(call, n_steady=5)
    except Exception as exc:  # a backend/compiler failure is itself a valid spike datum
        print(f"RESULT wide_scan nbins={NBINS} backend=gtfn_cpu FAILED: {exc!r}")
        return False

    got = np.stack([o.asnumpy() for o in outs])
    assert np.allclose(got, expected, rtol=1e-12), "wide scan gtfn_cpu wrong"
    print(
        f"RESULT wide_scan nbins={NBINS} backend=gtfn_cpu "
        f"first={first:.1f}s steady={steady * 1e3:.1f}ms"
    )
    return True


def measure_embedded(op, ncells: int) -> bool:
    """embedded measurement at a given ncells (common.NLEV levels always).
    Used both for the default small-NCELLS calibration ladder and for the
    opt-in full-scale (ncells=common.NCELLS) attempt.
    """
    rng = np.random.default_rng(11)
    q_np = rng.uniform(0.0, 1e-3, size=(NBINS, ncells, common.NLEV))
    rho_np = rng.uniform(0.8, 1.2, size=(ncells, common.NLEV))
    inputs = [common.make_field(q_np[b]) for b in range(NBINS)]
    rho = common.make_field(rho_np)
    expected = numpy_reference(q_np, rho_np, 0.5)
    outs = tuple(common.zeros_field(shape=(ncells, common.NLEV)) for _ in range(NBINS))

    def call() -> None:
        op(*inputs, rho, 0.5, out=outs, offset_provider={})

    try:
        first, steady = common.time_first_and_steady(call, n_steady=5)
    except Exception as exc:  # a backend/compiler failure is itself a valid spike datum
        print(f"RESULT wide_scan nbins={NBINS} backend=embedded ncells={ncells} FAILED: {exc!r}")
        return False

    got = np.stack([o.asnumpy() for o in outs])
    assert np.allclose(got, expected, rtol=1e-12), f"wide scan embedded ncells={ncells} wrong"
    print(
        f"RESULT wide_scan nbins={NBINS} backend=embedded ncells={ncells} "
        f"first={first:.2f}s steady={steady * 1e3:.2f}ms"
    )
    return True


def main() -> None:
    src = gen_source(NBINS)
    op = common.load_generated_operator(src, f"gen_sed_{NBINS}", f"_sed_{NBINS}")

    run_perturbation_check()

    backends = active_backends()
    ok = True

    if "gtfn_cpu" in backends:
        ok &= measure_gtfn(op)

    if "embedded" in backends:
        if os.environ.get("AMPS_SPIKE_C_FULL_EMBEDDED"):
            ok &= measure_embedded(op, common.NCELLS)
        else:
            for ncells in EMBEDDED_CALIBRATION_NCELLS:
                ok &= measure_embedded(op, ncells)

    print("SPIKE C: PASS" if ok else "SPIKE C: PARTIAL (see RESULT ... FAILED lines above)")


if __name__ == "__main__":
    main()

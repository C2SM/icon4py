# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Warm-phase operator-split host-loop SKELETON, transcribed from
`cal_micro_tendency` / `ifc_cloud_micro`, per
docs/superpowers/facts/m2/micro-tendency-orchestration.md ("G1" in comments
below), §1/§2, restricted to the warm-phase call subset G1's own
"Orchestration summary for the M2 host loop" section identifies:

    keep coalescence(rain,rain), hydrodyn_breakup(rain), warm activation
    (cal_aptact_var8_*), vapor_deposition(rain), and the full refresh/
    repair/update_group_all/cal_dmtend_scale skeleton; drop the four
    [ICE-ONLY] calls (ice-ice aggregation, ice-rain riming,
    hydrodyn_breakup(ice), Ice_Nucleation1/2, vapor_deposition(ice)).
    melting_shedding and the mv_ice2liq (level>=6) block are ice-sourced
    and only fire when flagp_s>0.

SKELETON (M2a Task 1) + `diag_pq` liquid-branch wiring (M2a Task 2): this
module delivers the col_loop x n_step_cl / vap_loop x n_step_vp substep
structure, the dt_cl/dt_vp setup (G1 §3), and the refresh preamble (G1 §1:
`update_mesrc` -> `diag_t` -> `update_airgroup` -> `diag_pq`, duplicated 4x
in the Fortran) collapsed into ONE function, `_refresh_state`. The warm-phase
PROCESS kernels are explicitly OUT of scope here and land in later tasks:

* Task 2 (DONE): `diag_pq` for the liquid (rain) group --
  `core.liquid_diag.diag_pq_liquid` (`core/liquid_diag.py`).
  `_refresh_state` now calls it directly whenever it is given a `config`/
  `luts` (both optional, default `None` -- see `_refresh_state`'s own
  docstring for why), populating `WarmLoopState.diag`.
  `update_airgroup` itself (imported by name in G1 §1 but not quoted
  verbatim anywhere in G1) still has no M1 port -- see below.
* Task 3: collision-substep warm physics (`coalescence(rain,rain)`,
  `hydrodyn_breakup(rain)`) -- NOT modeled at all in this skeleton (no
  stub hook exists for them here); Task 3 is expected to splice its calls
  into `run_warm_micro_tendency`'s col_loop body directly.
* Task 4 (DONE): `_activation` (CCN activation, `cal_aptact_var8_kc04dep`,
  G1 §1 vap_loop) -- `core.activation.activate_and_advance_vapor`
  (`core/activation.py`). Requires `state.diag` (Task 2's `diag_pq_liquid`
  output); `_refresh_state` always populates it immediately before this
  hook fires.
* Task 5 (DONE): `_vapor_deposition_liquid` (vapor deposition on rain,
  `vapor_deposition(CM%rain,...)`, G1 §1 vap_loop) --
  `core.vapor_deposition.vapor_deposition_liquid` (`core/
  vapor_deposition.py`). Requires `state.diag` (Task 2's
  `diag_pq_liquid` output, deliberately STALE relative to Task 4's own
  activation-added droplets -- see that module's own docstring item 2);
  `_refresh_state` always populates it immediately before this hook
  fires, exactly as for `_activation`.
* Task 6 (DONE): `_repair` (repair, G1 §1, called once per col_loop
  iteration and once per vap_loop iteration with DIFFERENT algorithms,
  not just different tendency masks) -- `core.repair.repair_liquid` (af_col,
  per-bin mass/concentration non-negativity) and `core.repair.repair_vapor`
  (af_vap, point-level vapor-supply closure protecting `state.thermo.qvv`)
  (`core/repair.py`, `docs/superpowers/facts/m2/sedimentation-terminalvel.md`
  ("G5") §4 + `cal_mass_budget_vapor`, `mod_amps_check.F90:2034-3145`, read
  directly). A code-review-caught earlier draft of this hook incorrectly
  called `repair_liquid` for BOTH phases; see `core/repair.py`'s own module
  docstring for the two algorithms' actual differences.

`update_group_all`/`cal_dmtend_scale` (G1 §4c, tendency bookkeeping) are
likewise NOT modeled here -- no M1 port exists for them and they are not
named in this task's brief-scoped deliverables (`WarmLoopState`,
`run_warm_micro_tendency`, `ifc_warm`, the three stub hooks above); a
future task wires them in alongside Task 3's collision-substep physics.

`update_airgroup` (`class_AirGroup.F90`, imported by name in G1 §1 but not
quoted verbatim anywhere in G1) has no M1 port either; `_refresh_state`
only refreshes `ThermoState.tv` (via M1's `core.thermo.diag_t`, F1 §5) and
leaves `thv`/`piv`/`qvv` -- which `update_airgroup` would also refresh in
the Fortran -- stale. This is a documented, deliberate simplification: it
does not affect this task's own test surface (substep/refresh bookkeeping,
dt_cl/dt_vp exactness, diag_t reproduction), and is closed out whenever
`update_airgroup` itself gets a real port.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.activation import (
    activate_and_advance_vapor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import (
    LiquidDiag,
    diag_pq_liquid,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import AmpsLuts
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import (
    PackedAmpsState,
    ScaleRawState,
    ScaleTendencies,
    get_thermo_prop,
    pack_scale_to_amps,
    unpack_amps_to_scale,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.repair import (
    repair_liquid,
    repair_vapor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.thermo import diag_t as _diag_t_f1
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.vapor_deposition import (
    vapor_deposition_liquid,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# WarmLoopState -- M1 LiquidState/AerosolState/ThermoState + the per-column
# "air-group" scalars G1 §1 carries alongside them (`thil`, `qtp`: plain
# `cal_micro_tendency(...,qtp,thil,...)` arguments, held FIXED across every
# substep -- `diag_t` re-diagnoses T from the fixed `thil` each refresh, it
# never mutates `thil` itself; `mes_rc`: the hydrometeor-phase flag G1 §4a's
# `update_mesrc` produces, refreshed on every `_refresh_state` call).
# No IceState: this is the WARM-phase bundle only (see module docstring).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmLoopState:
    """One packed column-vector's warm-phase microphysics state: the M1
    `ThermoState`/`LiquidState`/`AerosolState` bundles plus the per-column
    scalars G1 §1 threads through `cal_micro_tendency` alongside them.

    `thil`/`qtp` are HELD FIXED through `run_warm_micro_tendency` (matching
    G1 §1: `thil(*)`/`qtp(*)` are plain incoming arguments, never
    reassigned in the quoted Fortran; only `ag%TV(n)%T` -- here
    `thermo`'s `tv` field -- is re-diagnosed from them each refresh).
    `mes_rc` is NOT fixed: `_refresh_state` recomputes it every call (G1
    §4a `update_mesrc`, warm-only reduction, see `_update_mesrc_warm`).

    All arrays share one `npoints` (validated in `__post_init__`, mirroring
    `driver/box.py`'s `BoxCase` pattern) -- EXCEPT `diag`, whose shape is
    `(nbins, npoints)` (one entry per bin, not per column); not included in
    the `npoints` cross-check below.

    `diag`: `core.liquid_diag.LiquidDiag`, `diag_pq`'s liquid-branch output
    (M2a Task 2), refreshed by `_refresh_state` whenever it is called with a
    `config`/`luts` (see that function's docstring); `None` before the
    first such refresh (e.g. right after `WarmLoopState.__init__`, or in
    tests that call `_refresh_state(state)` with no `config`/`luts` to
    probe only the `diag_t` refresh -- see `test_warm_loop.py`).
    """

    thermo: ThermoState
    liquid: LiquidState
    aerosol: AerosolState
    thil: np.ndarray  # (npoints,) theta_il, G1 §1 `thil(*)`
    qtp: np.ndarray  # (npoints,) total water mixing ratio, G1 §1 `qtp(*)`
    mes_rc: np.ndarray  # (npoints,) hydrometeor phase flag, G1 §4a
    diag: LiquidDiag | None = None  # (nbins, npoints) per field; see above

    def __post_init__(self) -> None:
        npoints = {
            "thermo": self.thermo.npoints,
            "liquid": self.liquid.npoints,
            "aerosol": self.aerosol.npoints,
            "thil": self.thil.shape[0],
            "qtp": self.qtp.shape[0],
            "mes_rc": self.mes_rc.shape[0],
        }
        if len(set(npoints.values())) != 1:
            raise ValueError(
                f"WarmLoopState fields must share the same npoints (single packed "
                f"column-vector); got {npoints}"
            )


# ---------------------------------------------------------------------------
# Refresh preamble (G1 §1: update_mesrc -> diag_t -> update_airgroup ->
# diag_pq), collapsed into ONE function -- see module docstring.
# ---------------------------------------------------------------------------


def _update_mesrc_warm(thermo: ThermoState, liquid: LiquidState) -> np.ndarray:
    """`update_mesrc` (G1 §4a, `class_Group.F90:11150-11220`), warm-only
    reduction: no ice group exists in `WarmLoopState`, so `M_ts(n)` (total
    ice mass) is identically 0, collapsing the Fortran's 5-value
    `mes_rc in {0,1,2,3,4}` space down to the 3 reachable values here
    (0=no water, 1=vapor only, 2=rain; 3=ice-only and 4=mixed are
    unreachable without an ice group).

    `M_tr(n)` sums the RAW `rmt_q` PPV slot over bins (matching G1 §4a's
    `M_tr(n)=M_tr(n)+gr%MS(i,n)%mass(rmt)` verbatim -- no aerosol-mass
    subtraction here; that only happens in `diag_t`'s own `qr_0`
    accumulation, `_rain_specific_humidity` below). `M_v(n)` is
    `qvv*moist_denv` (G1 §4a: `M_v(n) = ag%tv(n)%rv*ag%tv(n)%den`).

    `moist_denv` is SI kg/m^3 (`state.py`'s own UNIT CONTRACT note on
    `ThermoProp.moist_denv`), UNCONVERTED here -- deliberately: `m_v` only
    ever feeds the `m_tot <= 0.0` sign check below, and `qvv`/`moist_denv`
    are both non-negative, so the check is scale-invariant regardless of
    which unit system the density is expressed in. This is the ONE
    documented exception to that field's "CGS consumers convert" rule (the
    other `core.liquid_diag`/`core.activation` consumers of `moist_denv`
    DO need the conversion -- see `state.py`'s own UNIT CONTRACT note).
    """
    lp = index_maps.LiquidPPV
    m_v = get_thermo_prop(thermo, ThermoProp.qvv) * get_thermo_prop(thermo, ThermoProp.moist_denv)
    m_tr = np.sum(liquid.values[lp.rmt_q.py_idx], axis=(0, 1))
    m_tot = m_v + m_tr
    mes_rc = np.where(m_tot <= 0.0, 0, np.where(m_tr > 0.0, 2, 1))
    return mes_rc.astype(np.int64)


def _rain_specific_humidity(liquid: LiquidState, mes_rc: np.ndarray) -> np.ndarray:
    """`qr_0` accumulation, `diag_t`'s rain branch (G1 §4b,
    `mod_amps_core.F90:12449-12550`), restricted to `mes_rc in {2,4}` (4,
    mixed, is unreachable in the warm-only `WarmLoopState`, kept in the
    mask for fidelity to G1's own guard). `LiquidState`'s `rmt_q`/`rmat_q`
    PPV slots are ALREADY per-unit-moist-air-mass (see `core/packing.py`'s
    `pack_scale_to_amps` and `core/thermo.py`'s `diag_t` docstring: "qr, qi
    ... already divided by density") -- unlike G1 §4b's own `g%MS%mass`
    space, which needs an explicit `/ag%TV(n)%den` G1 §4b performs inline;
    that division is therefore NOT repeated here.
    """
    lp = index_maps.LiquidPPV
    rmt = liquid.values[lp.rmt_q.py_idx]
    rmat = liquid.values[lp.rmat_q.py_idx]
    contrib = np.maximum(0.0, rmt - rmat)
    qr = np.sum(contrib, axis=(0, 1))
    is_rain = (mes_rc == 2) | (mes_rc == 4)
    return np.where(is_rain, qr, 0.0)


def _refresh_state(
    state: WarmLoopState, config: AmpsConfig | None = None, luts: AmpsLuts | None = None
) -> WarmLoopState:
    """The refresh preamble, G1 §1: `update_mesrc` -> `diag_t` ->
    `update_airgroup` -> `diag_pq`, collapsed into ONE function reused at
    every refresh point (`run_warm_micro_tendency`'s col-loop `it_cl>1`
    block, vap-loop head, and final post-loop refresh) instead of G1's
    4x-duplicated inline block. See the module docstring for exactly which
    of the four preamble steps are real here (`update_mesrc`, `diag_t`,
    `diag_pq`'s liquid branch) vs. unmodeled (`update_airgroup`).

    `config`/`luts`: OPTIONAL, default `None`. When both are given, `diag_pq`
    (`core.liquid_diag.diag_pq_liquid`, M2a Task 2) is run and its result
    stored on the returned state's `diag` field; when either is omitted,
    `diag_pq` is skipped and `diag` is carried through unchanged (`None` if
    never populated). This keeps `_refresh_state(state)` -- no config/luts
    -- valid for callers that only care about the `update_mesrc`/`diag_t`
    refresh (e.g. `test_warm_loop.py`'s `TestRefreshStateDiagT`, predating
    Task 2); `run_warm_micro_tendency`/`ifc_warm` (below) always pass both.
    """
    mes_rc = _update_mesrc_warm(state.thermo, state.liquid)
    qr = _rain_specific_humidity(state.liquid, mes_rc)
    # ThermoProp.ptotv is SI Pa (state.py's own UNIT CONTRACT note on that
    # enum member); core.thermo.diag_t is CGS (AmpsConst.p00=1e6 dyn/cm^2,
    # matching test_thermo.py's own `p_cgs = pt_pa * 10.0` round-trip
    # precedent) -- convert at this, the point of use, not upstream.
    p_cgs = get_thermo_prop(state.thermo, ThermoProp.ptotv) * 10.0
    t_new, _ierror1 = _diag_t_f1(state.thil, p_cgs, qr, 0.0)

    thermo_values = state.thermo.values.copy()
    tv_prop_idx = list(ThermoState.PROPS).index(ThermoProp.tv)
    thermo_values[tv_prop_idx, 0, 0, :] = t_new
    new_thermo = ThermoState(values=thermo_values)

    refreshed = dataclasses.replace(state, thermo=new_thermo, mes_rc=mes_rc)

    if config is not None and luts is not None:
        diag = diag_pq_liquid(refreshed.liquid, refreshed.thermo, config, luts)
        refreshed = dataclasses.replace(refreshed, diag=diag)

    return refreshed


# ---------------------------------------------------------------------------
# Process hooks -- Tasks 4/5/6. `_activation` (Task 4), `_vapor_deposition_
# liquid` (Task 5), and `_repair` (Task 6) are all now implemented.
# ---------------------------------------------------------------------------


def _activation(
    state: WarmLoopState, config: AmpsConfig, dt_vp: float, luts: AmpsLuts
) -> WarmLoopState:
    """CCN activation (G1 §1 vap_loop: `cal_aptact_var8_vec`, `act_type==2`,
    or `cal_aptact_var8_kc04dep`, the `else`/default branch cloudlab's own
    `act_type=1` selects) -- M2a Task 4's `core.activation.
    activate_and_advance_vapor` (`cal_aptact_var8_kc04dep`, KC04-deposition
    variant, matching cloudlab's `act_type=1`).

    Requires `state.diag` (populated by `_refresh_state(state, config,
    luts)`, always called immediately before this hook in
    `run_warm_micro_tendency`'s own vap-loop head) -- raises `ValueError`
    naming the missing precondition if called directly on a state that
    skipped that refresh (e.g. a bare `WarmLoopState(...)`, `diag=None` by
    default).
    """
    if state.diag is None:
        raise ValueError(
            "_activation requires state.diag (core.liquid_diag.LiquidDiag) to be "
            "populated -- call _refresh_state(state, config, luts) first "
            "(run_warm_micro_tendency's own vap-loop head always does, immediately "
            "before this hook)."
        )
    liquid, aerosol, thermo = activate_and_advance_vapor(
        state.liquid, state.aerosol, state.thermo, config, dt_vp, luts, state.diag
    )
    return dataclasses.replace(state, liquid=liquid, aerosol=aerosol, thermo=thermo)


def _vapor_deposition_liquid(
    state: WarmLoopState, config: AmpsConfig, dt_vp: float, luts: AmpsLuts
) -> WarmLoopState:
    """Vapor deposition on rain (G1 §1 vap_loop: `vapor_deposition(CM%rain,
    ...)`, guarded by `micexfg(6)==1 .and. flagp_r>0`) -- M2a Task 5's
    `core.vapor_deposition.vapor_deposition_liquid` (Chen-Lamb semidiscrete
    condensation/evaporation growth, the full `cal_lincubprms_vec`
    linear/cubic mass-space bin remap, and aerosol/vapor diversion on
    evaporation -- updates `state.liquid`, `state.aerosol`, AND
    `state.thermo` (net vapor exchanged this substep), matching
    `_activation`'s own 3-field update shape).

    `luts` is accepted for interface parity with the other two process
    hooks (`_activation`, `_repair`) but UNUSED -- `vapor_deposition_liquid`
    consumes `diag.vapdep_coef1`/`vapdep_coef2` (already fully computed by
    `diag_pq_liquid`, no LUT lookups of its own on the liquid path).

    Requires `state.diag` (populated by `_refresh_state(state, config,
    luts)`, always called immediately before this hook in
    `run_warm_micro_tendency`'s own vap-loop head) -- raises `ValueError`
    naming the missing precondition if called directly on a state that
    skipped that refresh, matching `_activation`'s own precondition check.
    """
    del luts
    if state.diag is None:
        raise ValueError(
            "_vapor_deposition_liquid requires state.diag (core.liquid_diag.LiquidDiag) to be "
            "populated -- call _refresh_state(state, config, luts) first "
            "(run_warm_micro_tendency's own vap-loop head always does, immediately "
            "before this hook)."
        )
    liquid, aerosol, thermo = vapor_deposition_liquid(
        state.liquid, state.aerosol, state.thermo, config, dt_vp, state.diag
    )
    return dataclasses.replace(state, liquid=liquid, aerosol=aerosol, thermo=thermo)


def _repair(state: WarmLoopState, config: AmpsConfig, phase: str) -> WarmLoopState:
    """`repair` (G1 §1: called once per col_loop iteration with tag
    `'af_col'`/masks `(.false.,.true.)`, and once per vap_loop iteration
    with tag `'af_vap'`/masks `(.true.,.false.)`) -- M2a Task 6's
    `core.repair.repair_liquid`/`core.repair.repair_vapor`.

    `phase`: `"collision"` (af_col) or `"vapor"` (af_vap), distinguishing
    the two G1 call sites -- these are TWO DIFFERENT algorithms in G5
    (`core/repair.py`'s own module docstring, and `repair_vapor`'s own
    docstring, spell out exactly how and why; a code-review-caught earlier
    draft of this function incorrectly called `repair_liquid` for both).
    `"collision"` -> `repair_liquid` (per-bin mass/concentration
    non-negativity closure, `cal_mass_budget_col`/`cal_con_budget_col`).
    `"vapor"` -> `repair_vapor` (point-level vapor-supply closure,
    `cal_mass_budget_vapor` -- protects `state.thermo.qvv` from going
    negative from over-condensation; never touches concentration).
    """
    if phase == "collision":
        liquid = repair_liquid(state.liquid, config)
        return dataclasses.replace(state, liquid=liquid)
    if phase == "vapor":
        liquid, thermo = repair_vapor(state.liquid, state.thermo, config)
        return dataclasses.replace(state, liquid=liquid, thermo=thermo)
    raise ValueError(f"_repair: phase must be 'collision' or 'vapor'; got {phase!r}")


# ---------------------------------------------------------------------------
# run_warm_micro_tendency -- the warm subset of cal_micro_tendency (G1 §1).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SubstepDts:
    """`dt_cl`/`dt_vp`, G1 §3 (`class_Cloud_Micro.F90:380-390`, verbatim):
    `dt_cl = dt_step/n_step_cl` (collision substep); `dt_vp =
    dt_cl/n_step_vp` (vapor substep -- a subdivision of the COLLISION
    substep, NOT of the full dynamic step: total vapor substeps per host
    step = `n_step_cl*n_step_vp`, each of length
    `dt_step/(n_step_cl*n_step_vp)`)."""

    dt_cl: float
    dt_vp: float


def _substep_dts(dt: float, config: AmpsConfig) -> _SubstepDts:
    dt_cl = dt / config.n_step_cl
    dt_vp = dt_cl / config.n_step_vp
    return _SubstepDts(dt_cl=dt_cl, dt_vp=dt_vp)


def run_warm_micro_tendency(
    state: WarmLoopState, config: AmpsConfig, dt: float, luts: AmpsLuts
) -> WarmLoopState:
    """Warm subset of `cal_micro_tendency` (G1 §1): the `col_loop1` (over
    `n_step_cl` collision substeps, dt=`dt_cl`) x `vap_loop` (over
    `n_step_vp` vapor substeps, dt=`dt_vp`) structure, with the refresh
    preamble (`_refresh_state`) at every G1-identified refresh point and
    the Task 4/5/6 process hooks (`_activation`, `_vapor_deposition_liquid`,
    `_repair` -- all implemented, Tasks 4/5/6) at their G1-identified call
    sites: `_activation` then `_vapor_deposition_liquid` inside the
    vap_loop body (G1 §1's own CCN-activation-then-vapor-deposition
    ordering), `_repair` once per col_loop iteration (before the vap_loop)
    and once per vap_loop iteration (after `_vapor_deposition_liquid`) --
    `update_group_all` is inlined-by-design (module docstring): each
    process hook already returns its own advanced `WarmLoopState`, so there
    is no separate "apply accumulated tendencies" step to model.

    Per G1 §1 (warm-only reduction, module docstring): the collision-substep
    physics (`coalescence(rain,rain)`, `hydrodyn_breakup(rain)`, Task 3) and
    `update_group_all`/`cal_dmtend_scale` bookkeeping are NOT modeled in
    this skeleton -- only the refresh + `_repair` bookkeeping G1 §1's
    `col_loop1` body performs unconditionally every iteration is here.

    Refresh call sites, exactly G1 §1's four (one dropped -- see below):
      1. col-loop `it_cl>1` block (906-916) -- conditional, `n_step_cl-1`
         times total.
      2. `mv_ice2liq` (`level_comp>=6 .and. it_cl==1`) preamble (932-953)
         -- ICE-SOURCED (moves melted ice into rain), DROPPED here: no ice
         group exists in `WarmLoopState` for it to move mass from.
      3. vap-loop head (1206-1214) -- unconditional, every
         `(it_cl, it_vp)` pair, `n_step_cl*n_step_vp` times total.
      4. final post-loop refresh (1374-1381) -- unconditional, once.
    """
    dts = _substep_dts(dt, config)

    for it_cl in range(1, config.n_step_cl + 1):
        if it_cl > 1:
            state = _refresh_state(state, config, luts)

        # Collision substep: G1 §1's warm-phase coalescence/breakup calls
        # (Task 3) are NOT modeled here -- see module/function docstrings.
        state = _repair(state, config, phase="collision")

        for _it_vp in range(1, config.n_step_vp + 1):
            state = _refresh_state(state, config, luts)
            state = _activation(state, config, dts.dt_vp, luts)
            state = _vapor_deposition_liquid(state, config, dts.dt_vp, luts)
            state = _repair(state, config, phase="vapor")

    state = _refresh_state(state, config, luts)
    return state


# ---------------------------------------------------------------------------
# ifc_warm -- warm path of ifc_cloud_micro (G1 §2).
# ---------------------------------------------------------------------------


def _reality_check_stub(state: WarmLoopState) -> WarmLoopState:
    """`reality_check` (`class_Cloud_Micro.F90`, called as
    `reality_check(CM,qtp,ID,JD,KD)` in G1 §2) -- bounds-checks `qtp`
    against a realistic physical range. No M1 port exists for this routine
    (not in this task's Reuse list); left as an identity pass-through, not
    a NotImplementedError-raiser (not one of the three named Task 4/5/6
    process hooks; `ifc_warm` needs to stay runnable end-to-end up to the
    first real stub it hits)."""
    return state


def ifc_warm(  # noqa: PLR0917 [too-many-positional-arguments]
    scale: ScaleRawState,
    thil: np.ndarray,
    qtp: np.ndarray,
    config: AmpsConfig,
    dt: float,
    luts: AmpsLuts,
    *,
    dens_t: np.ndarray,
    l_no_ice_heat: bool = False,
) -> ScaleTendencies:
    """Warm path of `ifc_cloud_micro` (G1 §2), reusing M1's
    `core.packing`/`state` throughout, per the task brief's 5-step
    sequence:

        ini (pack_scale_to_amps) -> reality_check (_reality_check_stub) ->
        refresh (_refresh_state) -> run_warm_micro_tendency ->
        return_output (unpack_amps_to_scale)

    `thil`/`qtp` are taken as direct arguments, matching G1 §2's own
    `ifc_cloud_micro(...,qtp,thil,...)` signature (both are plain
    pass-through arguments there too -- computed by an earlier,
    out-of-band `moistthermo2_scale` host-driver step, NOT inside
    `ifc_cloud_micro` itself, so `ifc_warm` does not compute them either).

    Does NOT model `check_water_apmass`/`update_terminal_vel` (G1 §2's
    other `ifc_cloud_micro` calls) -- out of the brief's 5-step sequence.
    The returned ice PPV state is `scale`'s own packed-before ice state,
    unchanged (this skeleton runs no ice process, so "ice after" ==
    "ice before" by construction).
    """
    packed_before = pack_scale_to_amps(scale, l_no_ice_heat=l_no_ice_heat)

    warm_state = WarmLoopState(
        thermo=packed_before.thermo,
        liquid=packed_before.liquid,
        aerosol=packed_before.aerosol,
        thil=np.asarray(thil, dtype=np.float64),
        qtp=np.asarray(qtp, dtype=np.float64),
        mes_rc=np.zeros(packed_before.thermo.npoints, dtype=np.int64),
    )
    warm_state = _reality_check_stub(warm_state)
    warm_state = _refresh_state(warm_state, config, luts)
    warm_state = run_warm_micro_tendency(warm_state, config, dt, luts)

    amps_after = PackedAmpsState(
        thermo=warm_state.thermo,
        liquid=warm_state.liquid,
        ice=packed_before.ice,
        aerosol=warm_state.aerosol,
        emoist_before=packed_before.emoist_before,
        liquid_heat_before=packed_before.liquid_heat_before,
        ice_heat_before=packed_before.ice_heat_before,
    )
    return unpack_amps_to_scale(
        scale,
        packed_before,
        amps_after,
        dens_t=np.asarray(dens_t, dtype=np.float64),
        dt=dt,
        l_no_ice_heat=l_no_ice_heat,
    )

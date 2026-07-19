# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Box-driver: `BoxCase` bundles one runnable single-column AMPS
microphysics case (thermo profile + initial liquid/ice/aerosol spectra +
`AmpsConfig` + timestep size/count); `run_box(case)` is the entry point a
real per-timestep driver loop uses -- implemented for the WARM path (M2a
Task 7: `implementations.warm_loop.run_warm_micro_tendency`, see `run_box`'s
own docstring for the exact bridge from `case`'s AMPS-packed state to a
`WarmLoopState`); a case carrying ice mass raises `NotImplementedError`
(M3 scope, the ice/mixed-phase call subset `implementations/warm_loop.py`
does not model -- see that module's own docstring).
`case_from_micro_record` IS implemented -- it is the M2 replay entry point,
turning one dumped `ref_data.MicroRecord` (phase=1, "pre") into a runnable
`BoxCase` so a captured reference column can be re-run standalone against
the ported Python/DSL physics (`run_box(case_from_micro_record(pre))`) and
diffed against that same record pair's phase=2 "post" `MicroRecord`
(`ref_data.RefDataset.micro_pairs()` gives both halves of the pair
together; `tests/amps/integration_tests/test_warm_replay.py` is the
per-call replay harness that does exactly this).

Scope note: a "box" run is single-column (`ncells=1` in every state
bundle's flattened `npoints` axis, per `state.py`'s `to_fields()`
convention) -- there is no horizontal transport to drive here, only the
per-column microphysical process tendencies (collision-coalescence, vapor
deposition, ice nucleation, ...) plus (optionally, later) sedimentation.

`MicroRecord` fact-gap affecting `case_from_micro_record`'s `ThermoState`
reconstruction: `AMPS_DUMP_micro` (the Fortran dump this record comes
from) captures `ptotvm, tvm, wbvm, qvvm, moist_denvm` directly, but never
`momv` (`w` at the half/MOMZ grid) -- that quantity is captured ONLY by
the SEPARATE `AMPS_DUMP_sed` dump, as `momz_col`
(`scale_atmos_phy_mp_amps.F90:5327/5358`; see `ref_data.py`'s own
module docstring for the full sed-input derivation notes). Since `momv`
is consumed only by sedimentation (not by the collision/vapor-deposition/
nucleation processes a "micro" phase covers), `case_from_micro_record`
defaults the reconstructed `ThermoState.momv` to zero and documents this
explicitly rather than silently leaving it wrong -- see that function's
own docstring.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import (
    SCALE_CPDRY,
    SCALE_PRE00,
    SCALE_RDRY,
    get_thermo_prop,
    moistthermo_mask,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.driver import ref_data
from icon4py.model.atmosphere.subgrid_scale_physics.amps.implementations import warm_loop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    IceState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


@dataclasses.dataclass(frozen=True)
class BoxResult:
    """`run_box`'s result bundle: the advanced state after `case.n_steps`
    warm-phase microphysics steps (see `run_box`'s own docstring for
    exactly which fields are genuinely advanced vs. carried through
    unchanged -- `final_ice` is always `case.ice`, untouched, since the
    warm path runs no ice process; `final_thermo`'s `thv`/`piv`/
    `moist_denv` are likewise carried through unchanged, a documented
    `implementations.warm_loop` scope gap, not new here). Does not (yet)
    carry tendency diagnostics alongside the final state -- a future task
    may extend this if a caller needs them."""

    final_thermo: ThermoState
    final_liquid: LiquidState
    final_ice: IceState
    final_aerosol: AerosolState


@dataclasses.dataclass(frozen=True)
class BoxCase:
    """One runnable single-column AMPS microphysics case: initial thermo
    profile + initial liquid/ice/aerosol spectra + the `AmpsConfig`
    governing which processes run + the run-time step size/count.

    Every state bundle's `npoints` must agree (single-column consistency,
    see module docstring) -- validated in `__post_init__`.
    """

    thermo: ThermoState
    liquid: LiquidState
    ice: IceState
    aerosol: AerosolState
    config: AmpsConfig
    dt: float
    n_steps: int

    def __post_init__(self) -> None:
        npoints = {
            "thermo": self.thermo.npoints,
            "liquid": self.liquid.npoints,
            "ice": self.ice.npoints,
            "aerosol": self.aerosol.npoints,
        }
        if len(set(npoints.values())) != 1:
            raise ValueError(
                f"BoxCase state bundles must share the same npoints (single-column "
                f"consistency); got {npoints}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive; got {self.dt}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive; got {self.n_steps}")


def run_box(
    case: BoxCase,
    *,
    luts: AmpsLuts | None = None,
    allow_shed_placeholder: bool = False,
    allow_dhf_placeholder: bool = False,
    allow_dep_placeholder: bool = False,
) -> BoxResult:
    """Run `case.n_steps` warm-phase microphysics steps of size `case.dt`
    on `case`'s single column and return the final state -- the WARM path
    only (M2a Task 7). A case carrying ice mass raises
    `NotImplementedError` (M3 scope, the ice/mixed-phase call subset
    `implementations/warm_loop.py` does not model -- see that module's own
    docstring).

    `allow_shed_placeholder`/`allow_dhf_placeholder`/`allow_dep_placeholder`:
    threaded straight through to every `implementations.warm_loop.
    _activation` call (all default `False`, this function's OWN prior
    behavior -- see `_activation`'s own docstring for the REAL-DATA finding
    that motivated exposing these here: a real cloudlab column, even from
    the nominally ice-free `warm` spin-up run, can still trip
    `activate_and_advance_vapor`'s unported DHF/classical-CNT-deposition
    ice-nucleation branches on cold, supersaturated LEVELS, raising
    `NotImplementedError` regardless of `case`'s own ice-mass guard above).
    A real-data caller (the replay harness, `case_from_micro_record`-driven
    smoke tests) sets these `True` to get past that known gap.

    Bridging `case` (whose `thermo`/`liquid`/`ice`/`aerosol` are ALREADY
    AMPS-packed state -- `case_from_micro_record`'s own docstring: `qrpvm`/
    `qipvm`/`qapvm` are direct copies into `LiquidState`/`IceState`/
    `AerosolState`, so `implementations.warm_loop.ifc_warm`'s
    `pack_scale_to_amps` step would be WRONG here, it expects raw
    pre-conversion SCALE state) onto
    `implementations.warm_loop.run_warm_micro_tendency` (which additionally
    needs `thil`/`qtp`, the per-column scalars G1 §1
    (`docs/superpowers/facts/m2/micro-tendency-orchestration.md`) threads
    through `cal_micro_tendency` -- `BoxCase` does not carry them;
    `case_from_micro_record`'s own docstring explicitly does not map the
    record's `trpv_thil`/`trpv_qtp` fields, leaving this the open bridge
    this function closes):

    1. Derive `thil`/`qtp` (and a cloudy-mask-passed `liquid`) via
       `core.packing.moistthermo_mask` (F4 SS3.2,
       `docs/superpowers/facts/m1/state-packing-si-cgs.md`:
       `moistthermo2_scale`'s own "qc/qr/qi bin partition + cloudy mask +
       thil/qtp diagnosis") from `case.thermo`/`liquid`/`ice` -- the SAME
       algorithm that produces the Fortran's own `trpv_thil`/`trpv_qtp` in
       the first place, not an approximation of it.
    2. If the derived `qi` (total ice mixing ratio) is nonzero anywhere,
       raise `NotImplementedError`: an ice-bearing/mixed-phase column is
       M3 scope -- `WarmLoopState` has no ice group to advance it with, so
       silently dropping nonzero ice mass would be wrong, not merely
       incomplete.
    3. Build a `WarmLoopState` from `case.thermo`/`case.aerosol` + the
       masked liquid + the derived `thil`/`qtp`, `mes_rc` seeded to 0, then
       run ONE `_refresh_state` call before the loop -- parity with
       `ifc_warm`'s own 5-step sequence (`reality_check -> _refresh_state
       -> run_warm_micro_tendency`, that function's own docstring), so
       `mes_rc`/`thermo.tv`/`state.diag` are populated from the start
       rather than left at their placeholder seed values until
       `run_warm_micro_tendency`'s own first internal refresh. Confirmed
       inconsequential to the FINAL result either way (`run_warm_micro_
       tendency`'s only step before its own first refresh is `_repair(
       phase="collision")`, which reads neither `mes_rc` nor `state.diag`;
       see `core/repair.py`) -- this exists for state-shape parity with
       `ifc_warm`-driven runs, not because it changes the physics.
    4. Call `run_warm_micro_tendency` `case.n_steps` times (one call per
       `case.dt`-sized step, matching `case_from_micro_record`'s own
       "n_steps ... one 'pre'->'post' step" convention for a replayed
       record pair).
    5. Return a `BoxResult` with the advanced `thermo`/`liquid`/`aerosol`
       and `case`'s OWN (untouched) `ice` -- matching `ifc_warm`'s own
       "ice after == ice before" convention (that function's docstring):
       this path runs no ice process, so ice cannot have changed.

    KNOWN GAP, carried from `implementations/warm_loop.py` (not introduced
    here): `update_airgroup` has no M1/M2 port (see that module's own
    docstring), so `final_thermo`'s `thv`/`piv`/`moist_denv` come back
    bit-identical to `case.thermo`'s own values -- NOT a reproduction of
    whatever the real Fortran's `update_airgroup` would have computed.
    Only `tv`/`qvv` (via `_refresh_state`'s `diag_t` call and the
    activation/vapor-deposition/repair process hooks) and `liquid`/
    `aerosol` are genuinely advanced. Callers comparing `BoxResult` against
    a real dumped "post" `MicroRecord` (`tests/amps/integration_tests/
    test_warm_replay.py`) must restrict the comparison to `tv`/`qvv`/
    `liquid`/`aerosol` for this reason -- comparing `thv`/`piv`/
    `moist_denv` would fail on this documented, pre-existing scope gap,
    not a bug in this function.

    Args:
        case: the `BoxCase` to run (must be ice-free to succeed; see above).
        luts: optional pre-loaded `AmpsLuts`. `core.lookup_tables.load_luts`
            reads two packaged `.npz` archives from disk on every call --
            callers making MANY `run_box` calls (e.g. a per-call replay
            harness iterating `ref_data.RefDataset.micro_pairs()`) should
            load once and pass the same `AmpsLuts` through rather than
            paying that cost per case. Defaults to loading internally.

    Raises:
        NotImplementedError: if `case` carries nonzero ice mass (the
            ice/mixed-phase path, M3 scope).
    """
    if luts is None:
        luts = load_luts()

    qv = get_thermo_prop(case.thermo, ThermoProp.qvv)
    th = get_thermo_prop(case.thermo, ThermoProp.thv)
    t = get_thermo_prop(case.thermo, ThermoProp.tv)
    mask = moistthermo_mask(case.liquid, case.ice, qv, th, t, nbhzcl=case.config.nbin_h)

    if np.any(mask.qi > 0.0):
        raise NotImplementedError(
            "run_box: case carries ice mass (moistthermo_mask's qi>0 for at least one "
            "column point) -- the ice/mixed-phase warm loop is M3 scope (see G1's "
            "[ICE-ONLY]/[MIXED] call-subset notes in "
            "docs/superpowers/facts/m2/micro-tendency-orchestration.md and "
            "implementations/warm_loop.py's own module docstring); only ice-free "
            "('warm') cases run today, via implementations.warm_loop."
        )

    state = warm_loop.WarmLoopState(
        thermo=case.thermo,
        liquid=mask.liquid,
        aerosol=case.aerosol,
        thil=mask.thp,
        qtp=mask.qtp,
        mes_rc=np.zeros(case.thermo.npoints, dtype=np.int64),
    )
    # Parity with `ifc_warm`'s own 5-step sequence (that function's own
    # docstring: "reality_check -> refresh -> run_warm_micro_tendency"):
    # one `_refresh_state` call before the loop, so `state.diag`/`mes_rc`/
    # `thermo.tv` are populated from the start rather than left at this
    # function's own placeholder seed values (`diag=None`, `mes_rc=0`)
    # until `run_warm_micro_tendency`'s own internal vap-loop-head refresh
    # first runs. Confirmed inconsequential to `run_warm_micro_tendency`'s
    # OWN output (its first action is `_repair(phase="collision")`, which
    # reads neither -- `core/repair.py`, `test_warm_loop.py`'s own
    # `test_repair_does_not_require_diag`) -- this call exists for
    # state-shape parity with `ifc_warm`, not because it changes the
    # result.
    state = warm_loop._refresh_state(state, case.config, luts)
    for _ in range(case.n_steps):
        state = warm_loop.run_warm_micro_tendency(
            state,
            case.config,
            case.dt,
            luts,
            allow_shed_placeholder=allow_shed_placeholder,
            allow_dhf_placeholder=allow_dhf_placeholder,
            allow_dep_placeholder=allow_dep_placeholder,
        )

    return BoxResult(
        final_thermo=state.thermo,
        final_liquid=state.liquid,
        final_ice=case.ice,
        final_aerosol=state.aerosol,
    )


def case_from_micro_record(
    rec: ref_data.MicroRecord,
    *,
    config: AmpsConfig | None = None,
    n_steps: int = 1,
) -> BoxCase:
    """Turn one dumped `MicroRecord` (must be phase=1, "pre") into a
    runnable `BoxCase` -- the M2 replay entry point: once `run_box` is
    implemented, `run_box(case_from_micro_record(pre))` is meant to be
    diffed against the SAME record pair's phase=2 "post" `MicroRecord`
    (`ref_data.RefDataset.micro_pairs()` gives you both halves together).

    Field mapping (`rec` -> `BoxCase`, `nmic` = the record's compressed
    column length -- see `MicroRecord`'s own docstring on `kmicvm`):

    * `liquid = LiquidState(values=rec.qrpvm[:len(LiquidPPV)])`, similarly
      `ice`/`aerosol` from `rec.qipvm`/`rec.qapvm` -- a PROPERTY-AXIS SLICE
      (a no-op when `rec.npr/npi/npa` already equal `len(LiquidPPV/IcePPV/
      AerosolPPV)` exactly, as every synthetic test fixture in this repo
      builds them), not always a direct copy: a REAL cluster dump instead
      has `rec.npr/npi/npa == len(LiquidPPV/IcePPV/AerosolPPV) + 2`.
      `scale_atmos_phy_mp_amps.F90:410/418/426` sets `npr/npi/npa =
      num_h_moments(i) + 2  ! two terminal velocity variables`, and
      confirms which 2: `AMPS_tv(k,i,j,ibr,m) = qrpv(npr-m+1,ibr,1,k)`
      (line 2238, `m=1,2`) reads exactly the LAST two property-axis slots
      as `m=1`: mass-weighted, `m=2`: con-weighted terminal velocity
      (`class_Cloud_Micro.F90:1784-1787` comment) -- diagnostics appended
      AFTER the true `len(...PPV)`-sized property vector, not consumed by
      `ifc_cloud_micro`'s activation/vapor-deposition/collision physics
      (which only ever reads Fortran indices `1..num_h_moments(i)`, i.e.
      this port's own `LiquidPPV`/`IcePPV`/`AerosolPPV` range) and not
      reproduced by this port's own `core.liquid_diag` terminal-velocity
      formulas either (they compute it fresh from bin mass/number, never
      read it as state) -- safe to drop for building a `BoxCase`.
      Confirmed empirically: `rec.npr/npi/npa - 2 == len(LiquidPPV/IcePPV/
      AerosolPPV)` on every inspected real `warm` record (`6-2=4`,
      `18-2=16`, `5-2=3`). Validated as EITHER `len(...)` (fixture
      convention) OR `len(...) + 2` (real-dump convention) -- raises
      `ValueError`, naming both the actual and both accepted counts, on
      anything else (a genuinely malformed record, not a fixture-vs-real
      shape difference this function is designed to absorb).
    * `rec.ncr/nci` must still be `1` (matches `state.py`'s liquid/ice
      `ncat` convention -- true of every real `warm` record inspected:
      `ncr=nci=1`, the category axis genuinely IS a singleton for these two
      groups). `rec.nca`, by contrast, is NOT required to be `1` --
      `scale_atmos_phy_mp_amps.F90:426-441`: `nca` is the config-selected
      AEROSOL category count (cloudlab's `ini_aerosol_prf=3` ->
      `nca=4`: "category 1,3,4 are CCN; category 2 is IN" -- matching
      `AmpsConfig.cloudlab()`'s own `dtype_a`/`fix_aerosol_type` 4-tuples,
      state.py's "ncat=1 everywhere" note was never true for aerosol in a
      real cloudlab run). `AerosolState` (a plain `(nprops, nbins, ncat,
      npoints)` numpy bundle, not one of `state.py`'s `to_fields()`-bound
      DSL paths -- the ncat==1 restriction lives THERE, not on the
      dataclass itself) carries the real `rec.nca`-sized category axis
      through unchanged; `core.activation.activate_and_advance_vapor`
      already consumes `aerosol.ncat` generically (`ncat = aerosol.ncat`,
      `dust_cat_idx = 1 if ncat >= 2 else None` -- the exact category-2-is-
      dust exclusion the Fortran comment above describes), so no further
      change was needed there to accept a real 4-category aerosol state.
    * `thermo.tv = rec.tvm`, `.qvv = rec.qvvm`, `.wbv = rec.wbvm` -- direct
      copies (all dumped, `Z_LOOP_01` lines 1643/1653/1664).
    * `thermo.ptotv = rec.ptotvm * 10.0`, `.moist_denv = rec.moist_denvm *
      1.0e-3` -- CGS-canonicalized, NOT direct copies: `rec.ptotvm`/`rec.
      moist_denvm` are the raw dumped SI values (`Z_LOOP_01` lines
      1641/1651), converted here to CGS (`ThermoState`'s own UNIT
      CONTRACT, `state.py`'s `ThermoProp` docstring) -- this function is
      one of `ThermoState`'s two ONLY producers, alongside `core.packing.
      _pack_thermo`.
    * `thermo.pbv = 0` (constant, `Z_LOOP_01` line 1650: `pbv(k) = 0.0_RP`).
    * `thermo.thv`, `.piv`, `.thetav` -- reconstructed via the SAME Exner
      relation `ref_data.py`'s sed-input derivation notes use, but WITHOUT
      that note's `thskinv`/`pgnd` caveat: here `ptotv`/`tv` are BOTH
      dumped directly (`ptotvm`/`tvm`), so no `QDRY`-dependent inversion is
      needed -- `thv = tv*(PRE00/ptotv_si)**(Rdry/CPdry)` (line 1644,
      exact, using the pre-conversion SI `ptotvm` -- `PRE00` is an SI
      constant), `piv = tv/thv*CPdry` (line 1647, exact), `thetav =
      thv*(1+0.61*qvv)` (line 1661, exact).
    * `thermo.momv = 0` -- FACT-GAP, not exact: `AMPS_DUMP_micro` never
      captures `momv`/MOMZ at all (only `AMPS_DUMP_sed` does, as
      `momz_col`); see module docstring. Defaulted to zero rather than
      guessed, since it is not consumed by any process a micro-phase
      record represents.
    * `config`: caller-supplied; defaults to `AmpsConfig.cloudlab()` since
      every M0 dump comes from the cloudlab reference run.
    * `dt = rec.dt` -- the record's own captured microphysics-substep `dt`.
    * `n_steps`: caller-supplied, defaults to 1 (one "pre"->"post" step,
      matching what the record pair itself spans).

    Raises:
        ValueError: if `rec.phase != ref_data.MicroRecord.PHASE_PRE`; if
            `rec.npr/npi/npa` aren't EITHER `len(LiquidPPV/IcePPV/
            AerosolPPV)` (fixture convention) OR that same count `+ 2`
            (the real-dump "+2 terminal-velocity slots" convention above --
            a genuinely malformed record, not a fixture-vs-real difference,
            fails this); or if `rec.ncr/nci` != 1 (state.py's liquid/ice
            `ncat` convention -- still exactly 1 in every real record
            inspected; `rec.nca` is deliberately NOT checked against 1, see
            above).
    """
    if rec.phase != ref_data.MicroRecord.PHASE_PRE:
        raise ValueError(
            f"case_from_micro_record expects a phase={ref_data.MicroRecord.PHASE_PRE} "
            f"('pre') record; got phase={rec.phase}"
        )
    # Two accepted npr/npi/npa conventions, per the docstring section above:
    # every synthetic fixture in this repo (test_ref_data.py's
    # `_make_pre_micro_record`, this module's own tests) builds `rec.npr ==
    # len(LiquidPPV)` exactly; a REAL cluster dump instead has `rec.npr ==
    # len(LiquidPPV) + 2` (2 trailing terminal-velocity diagnostic slots,
    # traced to scale_atmos_phy_mp_amps.F90:410/418/426 + the AMPS_tv
    # readback at line 2238). Both are accepted so this function keeps
    # working, unchanged, against every existing fixture while ALSO
    # accepting real dumps; anything else (e.g. a corrupted record) still
    # raises.
    n_liquid_props = len(LiquidState.PROPS)
    if rec.npr not in (n_liquid_props, n_liquid_props + 2):
        raise ValueError(
            f"rec.npr={rec.npr} not in {{len(LiquidPPV)={n_liquid_props}, "
            f"len(LiquidPPV)+2={n_liquid_props + 2}}} (the '+2' real-dump terminal-velocity-"
            "slots convention, see this function's own docstring); qrpvm's leading axis "
            "wouldn't match LiquidState.PROPS order"
        )
    if rec.ncr != 1:
        raise ValueError(
            f"rec.ncr={rec.ncr} != 1; qrpvm's category axis must be 1 (state.py's ncat "
            "convention -- true for liquid in every real record inspected)"
        )
    n_ice_props = len(IceState.PROPS)
    if rec.npi not in (n_ice_props, n_ice_props + 2):
        raise ValueError(
            f"rec.npi={rec.npi} not in {{len(IcePPV)={n_ice_props}, "
            f"len(IcePPV)+2={n_ice_props + 2}}} (the '+2' real-dump terminal-velocity-slots "
            "convention, see this function's own docstring); qipvm's leading axis wouldn't "
            "match IceState.PROPS order"
        )
    if rec.nci != 1:
        raise ValueError(
            f"rec.nci={rec.nci} != 1; qipvm's category axis must be 1 (state.py's ncat "
            "convention -- true for ice in every real record inspected)"
        )
    n_aero_props = len(AerosolState.PROPS)
    if rec.npa not in (n_aero_props, n_aero_props + 2):
        raise ValueError(
            f"rec.npa={rec.npa} not in {{len(AerosolPPV)={n_aero_props}, "
            f"len(AerosolPPV)+2={n_aero_props + 2}}} (the '+2' real-dump terminal-velocity-"
            "slots convention, see this function's own docstring); qapvm's leading axis "
            "wouldn't match AerosolState.PROPS order"
        )
    # rec.nca is intentionally NOT validated against 1 -- a real cloudlab
    # aerosol state genuinely carries 4 categories (see docstring above);
    # `nba` (aerosol bins) is likewise unchecked, exactly as before this
    # fix (AerosolState's shape is whatever rec.qapvm's own axes say).

    if config is None:
        config = AmpsConfig.cloudlab()

    # Property-axis slice: keep the first `len(...PPV)` entries, dropping
    # the trailing 2 terminal-velocity diagnostic slots this function's own
    # docstring documents (NOT part of the property vector `ifc_cloud_
    # micro`'s physics reads, and not reproduced from state by this port's
    # `core.liquid_diag` terminal-velocity formulas either -- see above).
    liquid = LiquidState(values=rec.qrpvm[:n_liquid_props])
    ice = IceState(values=rec.qipvm[:n_ice_props])
    aerosol = AerosolState(values=rec.qapvm[:n_aero_props])

    # `rec.ptotvm`/`rec.moist_denvm` are the RAW dumped SI values (`Z_LOOP_01`
    # lines 1641/1651: `ptotv(k) = PRES(k,i,j)` Pa, `moist_denv(k) =
    # DENS*factor_mxr1` kg/m^3) -- used in their native SI form for the
    # SI-Exner thv/piv/thetav derivation below (same as core/packing.py's
    # own _pack_thermo), then converted to CGS ONLY for the two ThermoState
    # fields that get stored, this being ThermoState's OTHER producer and
    # thus the other CGS-canonicalization point (state.py's ThermoProp UNIT
    # CONTRACT) -- `1 Pa = 10 dyn/cm^2`, `1 kg/m^3 = 1.0e-3 g/cm^3`.
    ptotv_si = rec.ptotvm
    tv = rec.tvm
    qvv = rec.qvvm
    # Same SCALE_RDRY/SCALE_CPDRY/SCALE_PRE00 Exner relation core/packing.py's
    # own _pack_thermo uses (Z_LOOP_01 lines 1644/1647/1650/1661) -- reused,
    # not re-derived.
    thv = tv * (SCALE_PRE00 / ptotv_si) ** (SCALE_RDRY / SCALE_CPDRY)  # line 1644
    piv = tv / thv * SCALE_CPDRY  # line 1647
    pbv = np.zeros_like(ptotv_si)  # line 1650
    thetav = thv * (1.0 + 0.61 * qvv)  # line 1661
    momv = np.zeros_like(ptotv_si)  # FACT-GAP -- see docstring above

    ptotv = ptotv_si * 10.0  # SI Pa -> CGS dyn/cm^2, ThermoState's own UNIT CONTRACT
    moist_denv = (
        rec.moist_denvm * 1.0e-3
    )  # SI kg/m^3 -> CGS g/cm^3, ThermoState's own UNIT CONTRACT

    nmic = rec.nmic
    thermo_values = np.empty((len(ThermoState.PROPS), 1, 1, nmic), dtype=np.float64)
    thermo_by_prop: dict[ThermoProp, np.ndarray] = {
        ThermoProp.ptotv: ptotv,
        ThermoProp.tv: tv,
        ThermoProp.thv: thv,
        ThermoProp.piv: piv,
        ThermoProp.pbv: pbv,
        ThermoProp.moist_denv: moist_denv,
        ThermoProp.qvv: qvv,
        ThermoProp.thetav: thetav,
        ThermoProp.wbv: rec.wbvm,
        ThermoProp.momv: momv,
    }
    for prop_idx, raw_prop in enumerate(ThermoState.PROPS):
        thermo_values[prop_idx, 0, 0, :] = thermo_by_prop[ThermoProp(int(raw_prop))]
    thermo = ThermoState(values=thermo_values)

    return BoxCase(
        thermo=thermo,
        liquid=liquid,
        ice=ice,
        aerosol=aerosol,
        config=config,
        dt=rec.dt,
        n_steps=n_steps,
    )

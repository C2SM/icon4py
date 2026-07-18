# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Box-driver skeleton: `BoxCase` bundles one runnable single-column AMPS
microphysics case (thermo profile + initial liquid/ice/aerosol spectra +
`AmpsConfig` + timestep size/count); `run_box(case)` is the entry point a
real per-timestep driver loop would use, but is not implemented here (see
its own docstring for the M2 wiring it needs); `case_from_micro_record`
IS implemented -- it is the M2 replay entry point, turning one dumped
`ref_data.MicroRecord` (phase=1, "pre") into a runnable `BoxCase` so a
captured reference column can eventually be re-run standalone against the
ported Python/DSL physics and diffed against that same record pair's
phase=2 "post" `MicroRecord` (`ref_data.RefDataset.micro_pairs()` gives
both halves of the pair together).

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
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import (
    SCALE_CPDRY,
    SCALE_PRE00,
    SCALE_RDRY,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.driver import ref_data
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    IceState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


@dataclasses.dataclass(frozen=True)
class BoxResult:
    """Placeholder result bundle for `run_box` -- shape is PROVISIONAL,
    to be finalized by whichever M2 task implements `run_box` (e.g. it may
    need to carry tendency diagnostics alongside the final state, not just
    the state itself). Not populated until `run_box` exists."""

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


def run_box(case: BoxCase) -> BoxResult:
    """Run `case.n_steps` microphysics steps of size `case.dt` on
    `case`'s single column and return the final state.

    NOT YET IMPLEMENTED -- this is the M2 wiring point. It needs the
    ported DSL/numpy microphysics process kernels (collision-coalescence,
    vapor deposition, ice nucleation, autoconversion, ... -- the processes
    `AmpsConfig.micexfg_array()` switches on) assembled into an actual
    per-timestep driver loop, none of which exist yet as of this task
    (M1 Task 8: only the typed reference-data reader and this skeleton are
    in scope). See the M1 plan's Task 8 brief and the M2 plan for the
    concrete process-kernel wiring this function is expected to call once
    those kernels exist.
    """
    raise NotImplementedError(
        "run_box is the M2 wiring point: it needs the ported DSL/numpy microphysics "
        "process kernels (collision-coalescence, vapor deposition, ice nucleation, ...) "
        "that later tasks provide, assembled into a per-timestep driver loop over "
        "case.n_steps. See box.py's module docstring and run_box's own docstring."
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

    * `liquid = LiquidState(values=rec.qrpvm)`, `ice = IceState(values=rec.qipvm)`,
      `aerosol = AerosolState(values=rec.qapvm)` -- direct copies: `qrpvm`/
      `qipvm`/`qapvm` are already shaped `(nprops, nbins, ncat, nmic)`,
      exactly `LiquidState`/`IceState`/`AerosolState`'s own convention
      (`state.py`'s `_BinnedState`), PROVIDED `rec.npr/npi/npa` equal
      `len(LiquidPPV)/len(IcePPV)/len(AerosolPPV)` -- validated here
      (raises `ValueError` on mismatch rather than silently
      misinterpreting the array axes).
    * `thermo.ptotv = rec.ptotvm`, `.tv = rec.tvm`, `.qvv = rec.qvvm`,
      `.moist_denv = rec.moist_denvm`, `.wbv = rec.wbvm` -- direct copies
      (all dumped, `Z_LOOP_01` lines 1638/1641/1653/1656/1664).
    * `thermo.pbv = 0` (constant, `Z_LOOP_01` line 1650: `pbv(k) = 0.0_RP`).
    * `thermo.thv`, `.piv`, `.thetav` -- reconstructed via the SAME Exner
      relation `ref_data.py`'s sed-input derivation notes use, but WITHOUT
      that note's `thskinv`/`pgnd` caveat: here `ptotv`/`tv` are BOTH
      dumped directly (`ptotvm`/`tvm`), so no `QDRY`-dependent inversion is
      needed -- `thv = tv*(PRE00/ptotv)**(Rdry/CPdry)` (line 1644, exact),
      `piv = tv/thv*CPdry` (line 1647, exact), `thetav = thv*(1+0.61*qvv)`
      (line 1661, exact).
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
        ValueError: if `rec.phase != ref_data.MicroRecord.PHASE_PRE`, or
            if `rec.npr/npi/npa` don't match the PPV enum lengths.
    """
    if rec.phase != ref_data.MicroRecord.PHASE_PRE:
        raise ValueError(
            f"case_from_micro_record expects a phase={ref_data.MicroRecord.PHASE_PRE} "
            f"('pre') record; got phase={rec.phase}"
        )
    if rec.npr != len(LiquidState.PROPS):
        raise ValueError(
            f"rec.npr={rec.npr} != len(LiquidPPV)={len(LiquidState.PROPS)}; "
            "qrpvm's leading axis wouldn't match LiquidState.PROPS order"
        )
    if rec.npi != len(IceState.PROPS):
        raise ValueError(
            f"rec.npi={rec.npi} != len(IcePPV)={len(IceState.PROPS)}; "
            "qipvm's leading axis wouldn't match IceState.PROPS order"
        )
    if rec.npa != len(AerosolState.PROPS):
        raise ValueError(
            f"rec.npa={rec.npa} != len(AerosolPPV)={len(AerosolState.PROPS)}; "
            "qapvm's leading axis wouldn't match AerosolState.PROPS order"
        )

    if config is None:
        config = AmpsConfig.cloudlab()

    liquid = LiquidState(values=rec.qrpvm)
    ice = IceState(values=rec.qipvm)
    aerosol = AerosolState(values=rec.qapvm)

    ptotv = rec.ptotvm
    tv = rec.tvm
    qvv = rec.qvvm
    # Same SCALE_RDRY/SCALE_CPDRY/SCALE_PRE00 Exner relation core/packing.py's
    # own _pack_thermo uses (Z_LOOP_01 lines 1644/1647/1650/1661) -- reused,
    # not re-derived.
    thv = tv * (SCALE_PRE00 / ptotv) ** (SCALE_RDRY / SCALE_CPDRY)  # line 1644
    piv = tv / thv * SCALE_CPDRY  # line 1647
    pbv = np.zeros_like(ptotv)  # line 1650
    thetav = thv * (1.0 + 0.61 * qvv)  # line 1661
    momv = np.zeros_like(ptotv)  # FACT-GAP -- see docstring above

    nmic = rec.nmic
    thermo_values = np.empty((len(ThermoState.PROPS), 1, 1, nmic), dtype=np.float64)
    thermo_by_prop: dict[ThermoProp, np.ndarray] = {
        ThermoProp.ptotv: ptotv,
        ThermoProp.tv: tv,
        ThermoProp.thv: thv,
        ThermoProp.piv: piv,
        ThermoProp.pbv: pbv,
        ThermoProp.moist_denv: rec.moist_denvm,
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

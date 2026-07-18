# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""SI (SCALE) <-> CGS-mixing-ratio (AMPS) state packing, transcribed from
`Z_LOOP_01` (pack) and the tendency/energy blocks (unpack), per
docs/superpowers/facts/m1/state-packing-si-cgs.md ("F4" below).

Three deliverables (numpy, array-in/array-out, mirroring the mp_amps driver
blocks in F4, per the task brief):

* `pack_scale_to_amps` -- F4 SS1 (`Z_LOOP_01`, lines 1625-1828): factor_mxr1
  mixing-ratio conversion, aerosol-mass-into-liquid/ice-mass addition, the
  x0.001 non-mass-PPV conversion applied to EXACTLY the indices F4 lists,
  Emoist "before" bookkeeping (both `l_no_ice_heat` branches).
* `unpack_amps_to_scale` -- F4 SS2 (tendency block, lines 2676-3010; energy
  block, lines 2305-2352): the RHOQ_t recipes, CPtot_t/CVtot_t, RHOE_t.
  `l_gaxis_version=1` (F4 SS2.4's "-- ORIGINAL" branch, including its
  `l_axis_limit` iag_q/icg_q clip-overwrite) is the only version implemented;
  2/3 raise `NotImplementedError` (F4 SS2.4's v2/v3 summaries are terse
  prose, not verbatim Fortran -- M5 scope per the task brief).
* `moistthermo_mask` -- F4 SS3.2 (`moistthermo2_scale`'s qc/qr/qi bin
  partition + cloudy-mask `micptr` + `qtp`/`thp` diagnosis, lines 1053-1117).
  SS3.3's iterative T-refinement/supersaturation-trigger machinery is a
  separate, later diagnostic NOT named in the task brief's scope ("micptrv
  cloudy-mask criterion + thil/qtp diagnosis") and is intentionally not
  ported here.

Shape convention: scalar-per-column SCALE fields are `(npoints,)`; per-bin
SCALE fields (`I_QL`, `I_QI`) are `(nbins, npoints)`; PPV property-vector
bundles are the Task 6 `LiquidState`/`IceState`/`AerosolState`/`ThermoState`
dataclasses (`(nprops, nbins, ncat, npoints)`, `ncat` required == 1 here, per
F4's own "every category loop is assume(d) to be 1" note already adopted by
`state.py`).

Scope note on the QTRC flat-array offset arithmetic (`ipr_qpr`/`ipi_qi` in
F4's verbatim Fortran): this module does NOT reconstruct "which flat QTRC
slot maps to which PPV property" -- that is an orthogonal SCALE-tracer-array
*wiring* concern (which QTRC index holds e.g. rime mass vs aggregate mass),
not one of the "equivalences" the task brief enumerates (factor_mxr1,
aerosol-into-mass addition, the x0.001 factor, Emoist). Callers instead
supply each SCALE-side raw quantity already keyed by the SAME
`LiquidPPV`/`IcePPV`/`AerosolPPV` property enum AMPS-side state already uses
(`ScaleRawState.liquid_ppv`/`ice_ppv`/`aerosol_ppv`, shaped like
`LiquidState`/`IceState`/`AerosolState` but holding PRE-conversion raw
values) -- this also sidesteps a genuine, F4-acknowledged ambiguity in the
Fortran `do ipi = imr_q, imc_q` ice loop trip count (F4's own text hedges
with "[rime, (agg,) crystal]"; the header's literal par_amps.F90 DEFAULT
values `imr_q=10, imc_q=11` given a literal reading would trip that loop
only twice, contradicting F4's own resolution -- via the `imt_q` aerosol-
addend offset arithmetic -- that it must trip three times (rime, agg,
crystal) to be internally consistent with "+4 is the 5th PPVI tracer =
imat_q". Both readings are documented in the task report as a fact-gap;
this module's per-property-keyed input design makes the question moot for
what it actually needs to compute.

NEEDS_CONTEXT (flagged, resolved via direct Fortran reads per the task's own
ground-truth allowance for a partially-quoted block's surrounding context --
see the task report for the full citation trail): F4 SS2.6 quotes
`CP_VAPOR`/`CP_WATER`/`CP_ICE`/`CV_VAPOR`/`CV_WATER`/`CV_ICE` and SS3.1
quotes `CPdry`/`Rdry`/`LHV0`/`LHS0`/bare `EPS` -- all as verbatim Fortran
identifiers, NONE with a defined numeric value anywhere in F4. Traced
directly against `scale_const.F90` and `scale_atmos_hydrometeor.F90`
(scale_amps repo) plus cloudlab's own `run.conf`/`restart_run.conf`
(`CONST_THERMODYN_TYPE = "SIMPLE"`, both files, identical), which pins the
concrete branch of `ATMOS_HYDROMETEOR_setup` used at runtime -- see the
`SCALE_*`/`CP_*`/`CV_*`/`EPS` module constants below for the exact citations
per value.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    AerosolState,
    IceState,
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# SCALE-side SI constants used by the mp_amps interface layer.
#
# DISTINCT from `core/constants.py`'s `AmpsConst` (CGS, AMPS-internal
# mod_amps_const.F90 values) and from `core/thermo.py`'s `CAL_THETAIL_*_MKS`
# constants (AMPS-native `cal_thetail`'s OWN local Fortran PARAMETERs in
# mod_amps_lib.F90, a DIFFERENT subroutine/purpose -- close in value to some
# of the constants below by physical coincidence, not interchangeable).
# These are scale_const.F90's `CONST_*` module parameters as imported,
# unmodified, into scale_atmos_phy_mp_amps.F90's `Z_LOOP_01`/tendency/energy
# blocks (F4 quotes the imported, renamed identifiers verbatim; this module
# traces them to their scale_const.F90 declarations -- see NEEDS_CONTEXT
# note in the module docstring).
# ---------------------------------------------------------------------------

SCALE_RDRY = 287.04  # J/K/kg; scale_const.F90:159 `CONST_Rdry = 287.04_RP`
SCALE_CPDRY = 1004.64  # J/K/kg; scale_const.F90:160 `CONST_CPdry = 1004.64_RP`
SCALE_PRE00 = 1.0e5  # Pa; SCALE `CONST_PRE00` (also cited by thermo.py's CAL_THETAIL_P0_PA)
SCALE_LHV0 = 2.501e6  # J/kg; scale_const.F90:82 `CONST_LHV0 = 2.501E+6_RP`
SCALE_LHS0 = 2.834e6  # J/kg; scale_const.F90:84 `CONST_LHS0 = 2.834E+6_RP`
SCALE_LHF0 = SCALE_LHS0 - SCALE_LHV0  # scale_const.F90:206 `CONST_LHF0 = CONST_LHS0 - CONST_LHV0`

# CP_VAPOR/CP_WATER/CP_ICE/CV_VAPOR/CV_WATER/CV_ICE: imported (unrenamed)
# from scale_atmos_hydrometeor.F90 into scale_atmos_phy_mp_amps.F90 (source
# line 1220-1226: `use scale_atmos_hydrometeor, only: CP_VAPOR, CP_WATER,
# CP_ICE, CV_VAPOR, CV_WATER, CV_ICE`). Their values are set once, at
# `ATMOS_HYDROMETEOR_setup`, branching on the namelist string
# `CONST_THERMODYN_TYPE` (scale_atmos_hydrometeor.F90:204-229). cloudlab's
# own run.conf/restart_run.conf both pin `CONST_THERMODYN_TYPE = "SIMPLE"`
# (run.conf:141, restart_run.conf:142), which resolves to the following
# assignments (scale_atmos_hydrometeor.F90:218-225, paraphrased so this
# block reads as documentation, not as commented-out Fortran):
#   CV_VAPOR takes CVvap; CP_VAPOR takes CPvap.
#   CV_WATER takes CVvap; CP_WATER takes CV_WATER's value (i.e. CVvap, NOT CPvap).
#   CV_ICE takes CVvap; CP_ICE takes CV_ICE's value (i.e. CVvap).
# where CPvap = CONST_CPvap = 1846.00 J/kg/K (scale_const.F90:69) and
# CVvap = CONST_CPvap - CONST_Rvap = 1846.00 - 461.50 = 1384.50 J/kg/K
# (scale_const.F90:68 `CONST_Rvap = 461.50_RP`; scale_const.F90:202
# `CONST_CVvap = CONST_CPvap - CONST_Rvap`). So under cloudlab's own
# thermodynamics choice, CP_WATER == CP_ICE == CV_WATER == CV_ICE == CV_VAPOR
# == CVvap; only CP_VAPOR differs (== CPvap).
_SCALE_CPVAP = 1846.00
_SCALE_CVVAP = _SCALE_CPVAP - 461.50  # 1384.50
CP_VAPOR = _SCALE_CPVAP
CV_VAPOR = _SCALE_CVVAP
CP_WATER = _SCALE_CVVAP
CV_WATER = _SCALE_CVVAP
CP_ICE = _SCALE_CVVAP
CV_ICE = _SCALE_CVVAP

# `EPS => CONST_EPS` (scale_atmos_phy_mp_amps.F90:25), used bare in F4 SS2.4's
# v1 `l_axis_limit` clip guard (`qipv(iacr_q,...) > EPS`). scale_const.F90:35
# declares `CONST_EPS = 1.E-16_RP` as the field's initializer, but
# `CONST_setup` (scale_const.F90:192) OVERWRITES it at runtime, before any
# physics runs: `CONST_EPS = epsilon(0.0_RP)` -- Fortran double-precision
# machine epsilon, 2^-52 = 2.220446049250313e-16, not the declared 1e-16.
# The guard is ~16 orders of magnitude below any physical axis-length-cubed
# value either way, so this has no behavioral consequence here; using the
# genuine runtime value for citation accuracy.
EPS = 2.220446049250313e-16  # double-precision machine epsilon, 2**-52

# F4 SS3.1 "for bin" thresholds (moistthermo2_scale), literal in F4 -- no
# gap here.
RRLMTB = 1.0e-22
RILMTB = 1.0e-22


# ---------------------------------------------------------------------------
# Raw (pre-conversion) SCALE-side input state.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScaleRawState:
    """Raw SCALE-side per-column quantities Z_LOOP_01 (F4 SS1) reads BEFORE
    the factor_mxr1/0.001 conversions -- the inputs to `pack_scale_to_amps`
    and (as the "before" reference) `unpack_amps_to_scale`.

    `liquid_ppv`/`ice_ppv`/`aerosol_ppv` are `LiquidState`/`IceState`/
    `AerosolState`-shaped, but hold RAW QTRC values keyed by the SAME PPV
    property enum (see module docstring for why): every slot is meaningful
    EXCEPT `LiquidPPV.rmt_q` and `IcePPV.imt_q`, which are pure pack OUTPUTS
    (computed from `ql`/`qi` plus the aerosol/melt addends -- F4 SS1.2/1.3)
    and are ignored on input. `IcePPV.imw_q`'s raw slot IS meaningful: it
    holds `I_QW` (melt water) directly, used both as `imw_q`'s own value
    and as one of `imt_q`'s addends.
    """

    dens: np.ndarray  # DENS(k,i,j), kg/m^3, (npoints,)
    qdry: np.ndarray  # QDRY(k,i,j), (npoints,)
    qv: np.ndarray  # QTRC(...,I_QV), (npoints,)
    pres: np.ndarray  # PRES(k,i,j), Pa, (npoints,)
    temp: np.ndarray  # TEMP(k,i,j), K, (npoints,)
    w: np.ndarray  # W(k,i,j), m/s, (npoints,)
    momz: np.ndarray  # MOMZ(k,i,j), (npoints,)
    ql: np.ndarray  # QTRC(...,I_QL+ibr-1): pure liquid water, (nbr, npoints)
    qi: np.ndarray  # QTRC(...,I_QI+ibi-1): pure ice, (nbi, npoints)
    liquid_ppv: LiquidState  # raw QPPVL (rmt_q slot ignored on input)
    ice_ppv: IceState  # raw QPPVI incl. imw_q=I_QW (imt_q slot ignored on input)
    aerosol_ppv: AerosolState  # raw QPPVA, all slots meaningful


@dataclasses.dataclass(frozen=True)
class PackedAmpsState:
    """`pack_scale_to_amps`'s output -- the AMPS-space (per-moist-air mixing
    ratio) state, F4 SS1, plus the Emoist "before" bookkeeping (F4 SS1.5).
    Also used, unmodified for its `.liquid`/`.ice`/`.aerosol`/`.thermo`
    fields, as `unpack_amps_to_scale`'s "after" (post-microphysics) AMPS
    state argument -- see that function's docstring.
    """

    thermo: ThermoState
    liquid: LiquidState
    ice: IceState
    aerosol: AerosolState
    emoist_before: np.ndarray  # Emoist(k,1), F4 SS1.5 line 1658 (+ LHF0 term unless l_no_ice_heat)
    #: liquid_heat(k,1)/ice_heat(k,1), F4 SS1.5's `l_no_ice_heat` branch --
    #: computed there but NOT consumed by `unpack_amps_to_scale`'s RHOE_t
    #: (F4 SS2.7: the liquid_heat/ice_heat alternative is "fully commented
    #: out" in the Fortran); kept here purely for pack-side fidelity to F4.
    liquid_heat_before: np.ndarray | None
    ice_heat_before: np.ndarray | None


def get_thermo_prop(thermo: ThermoState, prop: ThermoProp) -> np.ndarray:
    """Extract one named `ThermoProp` field as a plain `(npoints,)` array
    (`ThermoState` is pinned to `nbins=ncat=1`, see `state.py`)."""
    prop_index = list(ThermoState.PROPS).index(prop)
    return thermo.values[prop_index, 0, 0, :]


# ---------------------------------------------------------------------------
# pack_scale_to_amps -- F4 SS1 (Z_LOOP_01).
# ---------------------------------------------------------------------------

# Ice PPV property groupings for the generic per-property conversion loop
# (F4 SS1.3): "mass, plain /factor_mxr1" vs "non-mass, /factor_mxr1/0.001".
# imt_q (special: ice+melt+aerosol addend) and imw_q (special: also an imt_q
# addend, handled together with it) are excluded from both groups.
_ICE_MASS_PLAIN = (
    index_maps.IcePPV.imr_q,
    index_maps.IcePPV.ima_q,
    index_maps.IcePPV.imc_q,
    index_maps.IcePPV.imat_q,
    index_maps.IcePPV.imas_q,
    index_maps.IcePPV.imf_q,
)
_ICE_NONMASS = (
    index_maps.IcePPV.icon_q,
    index_maps.IcePPV.ivcs_q,
    index_maps.IcePPV.iacr_q,
    index_maps.IcePPV.iccr_q,
    index_maps.IcePPV.idcr_q,
    index_maps.IcePPV.iag_q,
    index_maps.IcePPV.icg_q,
    index_maps.IcePPV.inex_q,
)
_LIQUID_MASS_PLAIN = (index_maps.LiquidPPV.rmat_q, index_maps.LiquidPPV.rmas_q)
_LIQUID_NONMASS = (index_maps.LiquidPPV.rcon_q,)
_AEROSOL_MASS_PLAIN = (index_maps.AerosolPPV.amt_q, index_maps.AerosolPPV.ams_q)
_AEROSOL_NONMASS = (index_maps.AerosolPPV.acon_q,)


def _require_ncat_one(*bundles: LiquidState | IceState | AerosolState) -> None:
    for bundle in bundles:
        if bundle.ncat != 1:
            raise NotImplementedError(
                f"{type(bundle).__name__}: ncat={bundle.ncat} not supported (F4: every "
                "category loop is 'assume(d) to be 1')"
            )


def _require_gaxis_version_1(l_gaxis_version: int) -> None:
    if l_gaxis_version != 1:
        raise NotImplementedError(
            f"l_gaxis_version={l_gaxis_version} (2/3, F4 state-packing-si-cgs.md SS1.3/SS2.4 "
            "'METHOD 1'/'METHOD 2') is an M5 stub; only l_gaxis_version=1 ('-- ORIGINAL') is "
            "implemented here."
        )


def _pack_thermo(
    scale: ScaleRawState, factor_mxr1: np.ndarray, moist_denv: np.ndarray
) -> ThermoState:
    """F4 SS1.1, `Z_LOOP_01` thermo block (lines 1625-1672)."""
    ptotv = scale.pres
    tv = scale.temp
    thv = tv * (SCALE_PRE00 / ptotv) ** (SCALE_RDRY / SCALE_CPDRY)
    piv = tv / thv * SCALE_CPDRY
    pbv = np.zeros_like(tv)
    qvv = scale.qv / factor_mxr1
    thetav = thv * (1.0 + 0.61 * qvv)
    wbv = scale.w
    momv = scale.momz

    npoints = tv.shape[0]
    by_prop: dict[ThermoProp, np.ndarray] = {
        ThermoProp.ptotv: ptotv,
        ThermoProp.tv: tv,
        ThermoProp.thv: thv,
        ThermoProp.piv: piv,
        ThermoProp.pbv: pbv,
        ThermoProp.moist_denv: moist_denv,
        ThermoProp.qvv: qvv,
        ThermoProp.thetav: thetav,
        ThermoProp.wbv: wbv,
        ThermoProp.momv: momv,
    }
    values = np.empty((len(ThermoState.PROPS), 1, 1, npoints), dtype=np.float64)
    for prop_idx, raw_prop in enumerate(ThermoState.PROPS):
        values[prop_idx, 0, 0, :] = by_prop[ThermoProp(int(raw_prop))]
    return ThermoState(values=values)


def pack_scale_to_amps(
    scale: ScaleRawState,
    *,
    l_no_ice_heat: bool,
    l_gaxis_version: int = 1,
) -> PackedAmpsState:
    """SCALE `QTRC` -> AMPS `qrpv`/`qipv`/`qapv`, F4 SS1 (`Z_LOOP_01`)."""
    _require_gaxis_version_1(l_gaxis_version)
    _require_ncat_one(scale.liquid_ppv, scale.ice_ppv, scale.aerosol_ppv)

    dens = np.asarray(scale.dens, dtype=np.float64)
    qdry = np.asarray(scale.qdry, dtype=np.float64)
    qv = np.asarray(scale.qv, dtype=np.float64)
    factor_mxr1 = qdry + qv  # F4 SS1.1: QDRY(k,i,j) + QTRC(k,i,j,I_QV)
    moist_denv = dens * factor_mxr1  # == factor_mxr2 (F4: DENS*factor_mxr1)

    thermo = _pack_thermo(scale, factor_mxr1, moist_denv)

    # --- liquid spectrum, F4 SS1.2 ---
    lp = index_maps.LiquidPPV
    liquid_out = np.zeros_like(scale.liquid_ppv.values)
    for liquid_prop in _LIQUID_MASS_PLAIN:
        liquid_out[liquid_prop.py_idx] = scale.liquid_ppv.values[liquid_prop.py_idx] / factor_mxr1
    for liquid_prop in _LIQUID_NONMASS:
        liquid_out[liquid_prop.py_idx] = (
            scale.liquid_ppv.values[liquid_prop.py_idx] / factor_mxr1 / 0.001
        )
    ql_per_bin = scale.ql[:, None, :]  # (nbr, npoints) -> (nbr, 1, npoints)
    liquid_out[lp.rmt_q.py_idx] = (
        ql_per_bin + scale.liquid_ppv.values[lp.rmat_q.py_idx]
    ) / factor_mxr1

    # --- ice spectrum, F4 SS1.3 (l_gaxis_version=1 only) ---
    ip = index_maps.IcePPV
    ice_out = np.zeros_like(scale.ice_ppv.values)
    for ice_prop in _ICE_MASS_PLAIN:
        ice_out[ice_prop.py_idx] = scale.ice_ppv.values[ice_prop.py_idx] / factor_mxr1
    for ice_prop in _ICE_NONMASS:
        ice_out[ice_prop.py_idx] = scale.ice_ppv.values[ice_prop.py_idx] / factor_mxr1 / 0.001
    ice_out[ip.imw_q.py_idx] = scale.ice_ppv.values[ip.imw_q.py_idx] / factor_mxr1
    qi_per_bin = scale.qi[:, None, :]
    ice_out[ip.imt_q.py_idx] = (
        qi_per_bin + scale.ice_ppv.values[ip.imw_q.py_idx] + scale.ice_ppv.values[ip.imat_q.py_idx]
    ) / factor_mxr1

    # --- aerosol spectrum, F4 SS1.4 ---
    aerosol_out = np.zeros_like(scale.aerosol_ppv.values)
    for aerosol_prop in _AEROSOL_MASS_PLAIN:
        aerosol_out[aerosol_prop.py_idx] = (
            scale.aerosol_ppv.values[aerosol_prop.py_idx] / factor_mxr1
        )
    for aerosol_prop in _AEROSOL_NONMASS:
        aerosol_out[aerosol_prop.py_idx] = (
            scale.aerosol_ppv.values[aerosol_prop.py_idx] / factor_mxr1 / 0.001
        )

    # --- Emoist "before", F4 SS1.5 ---
    emoist_before = -SCALE_LHV0 * qv * dens
    liquid_heat_before: np.ndarray | None = None
    ice_heat_before: np.ndarray | None = None
    if not l_no_ice_heat:
        emoist_before = emoist_before + SCALE_LHF0 * np.sum(scale.qi, axis=0) * dens
    else:
        liquid_heat_before = SCALE_LHF0 * np.sum(scale.ql, axis=0) * dens
        ice_heat_before = SCALE_LHF0 * np.sum(scale.qi, axis=0) * dens

    return PackedAmpsState(
        thermo=thermo,
        liquid=LiquidState(values=liquid_out),
        ice=IceState(values=ice_out),
        aerosol=AerosolState(values=aerosol_out),
        emoist_before=emoist_before,
        liquid_heat_before=liquid_heat_before,
        ice_heat_before=ice_heat_before,
    )


# ---------------------------------------------------------------------------
# unpack_amps_to_scale -- F4 SS2 (RHOQ_t / CPtot_t / CVtot_t / RHOE_t).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScaleTendencies:
    """`unpack_amps_to_scale`'s output: `RHOQ_t` for every SCALE tracer
    slot, plus `CPtot_t`/`CVtot_t`/`RHOE_t` (F4 SS2). `d_liquid_ppv`'s
    `rmt_q` slot and `d_ice_ppv`'s `imt_q`/`imw_q` slots are unused
    placeholders left at 0 -- those tracers' tendencies are `dql`/`dqi`/
    `dqw` instead (mirroring `ScaleRawState`'s equivalent "ignored on input"
    slots, see that dataclass's docstring)."""

    dqv: np.ndarray  # RHOQ_t[I_QV], (npoints,)
    dql: np.ndarray  # RHOQ_t[I_QL+ibr-1], (nbr, npoints)
    dqi: np.ndarray  # RHOQ_t[I_QI+ibi-1], (nbi, npoints)
    dqw: np.ndarray  # RHOQ_t[I_QW+ibi-1], (nbi, npoints)
    d_liquid_ppv: LiquidState  # RHOQ_t for rmat_q/rmas_q/rcon_q (rmt_q slot unused)
    d_ice_ppv: IceState  # RHOQ_t for the remaining 13 ice PPV slots (imt_q/imw_q unused)
    d_aerosol_ppv: AerosolState  # RHOQ_t for amt_q/acon_q/ams_q
    cptot_t: np.ndarray  # (npoints,), F4 SS2.6
    cvtot_t: np.ndarray  # (npoints,), F4 SS2.6
    rhoe_t: np.ndarray  # (npoints,), F4 SS2.7


def _unpack_liquid(
    scale_before: ScaleRawState,
    amps_after: PackedAmpsState,
    dens: np.ndarray,
    moist_denv_after: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, LiquidState]:
    """F4 SS2.3: `I_QL` tendency (`dql`) + `rmat_q`/`rmas_q`/`rcon_q` RHOQ_t."""
    lp = index_maps.LiquidPPV
    rmt_after = amps_after.liquid.values[lp.rmt_q.py_idx]
    rmat_after = amps_after.liquid.values[lp.rmat_q.py_idx]
    ql_before = scale_before.ql[:, None, :]
    dql = ((rmt_after - rmat_after) * moist_denv_after - ql_before * dens) / dt

    d_liquid = np.zeros_like(scale_before.liquid_ppv.values)
    for prop in _LIQUID_MASS_PLAIN:
        d_liquid[prop.py_idx] = (
            amps_after.liquid.values[prop.py_idx] * moist_denv_after
            - scale_before.liquid_ppv.values[prop.py_idx] * dens
        ) / dt
    for prop in _LIQUID_NONMASS:
        d_liquid[prop.py_idx] = (
            amps_after.liquid.values[prop.py_idx] * moist_denv_after * 0.001
            - scale_before.liquid_ppv.values[prop.py_idx] * dens
        ) / dt
    return dql[:, 0, :], LiquidState(values=d_liquid)


def _apply_axis_limit_clip(  # noqa: PLR0917 [too-many-positional-arguments]
    d_ice: np.ndarray,
    scale_before: ScaleRawState,
    amps_after: PackedAmpsState,
    dens: np.ndarray,
    moist_denv_after: np.ndarray,
    dt: float,
) -> None:
    """F4 SS2.4 v1: `iag_q`/`icg_q` RHOQ_t is overwritten (in place, on
    `d_ice`) with the a/c-axis value instead of its own whenever the axis
    ratio exceeds 1."""
    ip = index_maps.IcePPV
    for axis_prop, cog_prop in ((ip.iacr_q, ip.iag_q), (ip.iccr_q, ip.icg_q)):
        axis_after = amps_after.ice.values[axis_prop.py_idx]
        cog_after = amps_after.ice.values[cog_prop.py_idx]
        mask = axis_after > EPS
        ratio = np.divide(cog_after, axis_after, out=np.zeros_like(axis_after), where=mask)
        clip = mask & (ratio > 1.0)
        clipped = (
            axis_after * moist_denv_after * 0.001
            - scale_before.ice_ppv.values[cog_prop.py_idx] * dens
        ) / dt
        d_ice[cog_prop.py_idx] = np.where(clip, clipped, d_ice[cog_prop.py_idx])


def _unpack_ice(  # noqa: PLR0917 [too-many-positional-arguments]
    scale_before: ScaleRawState,
    amps_after: PackedAmpsState,
    dens: np.ndarray,
    moist_denv_after: np.ndarray,
    dt: float,
    l_axis_limit: bool,
) -> tuple[np.ndarray, np.ndarray, IceState]:
    """F4 SS2.4: `I_QI`/`I_QW` tendencies (`dqi`/`dqw`) + the remaining ice
    PPV RHOQ_t (l_gaxis_version=1)."""
    ip = index_maps.IcePPV
    imt_after = amps_after.ice.values[ip.imt_q.py_idx]
    imw_after = amps_after.ice.values[ip.imw_q.py_idx]
    imat_after = amps_after.ice.values[ip.imat_q.py_idx]
    qi_before = scale_before.qi[:, None, :]
    dqi = ((imt_after - imat_after - imw_after) * moist_denv_after - qi_before * dens) / dt
    dqw = (imw_after * moist_denv_after - scale_before.ice_ppv.values[ip.imw_q.py_idx] * dens) / dt

    d_ice = np.zeros_like(scale_before.ice_ppv.values)
    for prop in _ICE_MASS_PLAIN:
        d_ice[prop.py_idx] = (
            amps_after.ice.values[prop.py_idx] * moist_denv_after
            - scale_before.ice_ppv.values[prop.py_idx] * dens
        ) / dt
    for prop in _ICE_NONMASS:
        d_ice[prop.py_idx] = (
            amps_after.ice.values[prop.py_idx] * moist_denv_after * 0.001
            - scale_before.ice_ppv.values[prop.py_idx] * dens
        ) / dt
    if l_axis_limit:
        _apply_axis_limit_clip(d_ice, scale_before, amps_after, dens, moist_denv_after, dt)

    return dqi[:, 0, :], dqw[:, 0, :], IceState(values=d_ice)


def _unpack_aerosol(
    scale_before: ScaleRawState,
    amps_after: PackedAmpsState,
    dens: np.ndarray,
    moist_denv_after: np.ndarray,
    dt: float,
) -> AerosolState:
    """F4 SS2.5: `amt_q`/`acon_q`/`ams_q` RHOQ_t."""
    d_aerosol = np.zeros_like(scale_before.aerosol_ppv.values)
    for prop in _AEROSOL_MASS_PLAIN:
        d_aerosol[prop.py_idx] = (
            amps_after.aerosol.values[prop.py_idx] * moist_denv_after
            - scale_before.aerosol_ppv.values[prop.py_idx] * dens
        ) / dt
    for prop in _AEROSOL_NONMASS:
        d_aerosol[prop.py_idx] = (
            amps_after.aerosol.values[prop.py_idx] * moist_denv_after * 0.001
            - scale_before.aerosol_ppv.values[prop.py_idx] * dens
        ) / dt
    return AerosolState(values=d_aerosol)


def _unpack_cptot_cvtot(  # noqa: PLR0917 [too-many-positional-arguments]
    scale_before: ScaleRawState,
    amps_after: PackedAmpsState,
    dens_new: np.ndarray,
    moist_denv_after: np.ndarray,
    qv_after: np.ndarray,
    dt: float,
    l_no_ice_heat: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """F4 SS2.6: `CPtot_t`/`CVtot_t`, this call's own contribution (not a
    running accumulator -- see `unpack_amps_to_scale`'s docstring)."""
    lp, ip = index_maps.LiquidPPV, index_maps.IcePPV
    rmt_after = amps_after.liquid.values[lp.rmt_q.py_idx]
    rmat_after = amps_after.liquid.values[lp.rmat_q.py_idx]
    ql_before = scale_before.ql[:, None, :]
    imt_after = amps_after.ice.values[ip.imt_q.py_idx]
    imw_after = amps_after.ice.values[ip.imw_q.py_idx]
    imat_after = amps_after.ice.values[ip.imat_q.py_idx]
    qi_before = scale_before.qi[:, None, :]

    dq_vapor = qv_after * moist_denv_after / dens_new - scale_before.qv
    cptot_t = CP_VAPOR * dq_vapor / dt
    cvtot_t = CV_VAPOR * dq_vapor / dt

    dq_liquid = np.sum(
        (rmt_after - rmat_after) * moist_denv_after / dens_new - ql_before, axis=(0, 1)
    )
    dq_liquid = dq_liquid + np.sum(
        imw_after * moist_denv_after / dens_new - scale_before.ice_ppv.values[ip.imw_q.py_idx],
        axis=(0, 1),
    )
    cptot_t = cptot_t + CP_WATER * dq_liquid / dt
    cvtot_t = cvtot_t + CV_WATER * dq_liquid / dt

    dq_ice = np.sum(
        (imt_after - imw_after - imat_after) * moist_denv_after / dens_new - qi_before,
        axis=(0, 1),
    )
    cp_ice, cv_ice = (CP_WATER, CV_WATER) if l_no_ice_heat else (CP_ICE, CV_ICE)
    cptot_t = cptot_t + cp_ice * dq_ice / dt
    cvtot_t = cvtot_t + cv_ice * dq_ice / dt
    return cptot_t, cvtot_t


def _unpack_rhoe_t(  # noqa: PLR0917 [too-many-positional-arguments]
    packed_before: PackedAmpsState,
    amps_after: PackedAmpsState,
    moist_denv_after: np.ndarray,
    qv_after: np.ndarray,
    dt: float,
    l_no_ice_heat: bool,
) -> np.ndarray:
    """F4 SS2.7: Emoist "after" + `RHOE_t`."""
    ip = index_maps.IcePPV
    emoist_after = -SCALE_LHV0 * qv_after * moist_denv_after
    if not l_no_ice_heat:
        imt_after = amps_after.ice.values[ip.imt_q.py_idx]
        imw_after = amps_after.ice.values[ip.imw_q.py_idx]
        imat_after = amps_after.ice.values[ip.imat_q.py_idx]
        ice_pure_after = imt_after - imw_after - imat_after
        emoist_after = emoist_after + SCALE_LHF0 * np.sum(
            ice_pure_after * moist_denv_after, axis=(0, 1)
        )
    return (emoist_after - packed_before.emoist_before) / dt


def unpack_amps_to_scale(
    scale_before: ScaleRawState,
    packed_before: PackedAmpsState,
    amps_after: PackedAmpsState,
    *,
    dens_t: np.ndarray,
    dt: float,
    l_no_ice_heat: bool,
    l_gaxis_version: int = 1,
    l_axis_limit: bool = True,
) -> ScaleTendencies:
    """AMPS post-call state -> SCALE `RHOQ_t`/`CPtot_t`/`CVtot_t`/`RHOE_t`,
    F4 SS2. `amps_after` is a `PackedAmpsState` from AFTER the (out-of-scope)
    AMPS microphysics call ran -- only its `.liquid`/`.ice`/`.aerosol` and
    `.thermo`'s `qvv`/`moist_denv` fields are read; `.emoist_before`/
    `.liquid_heat_before`/`.ice_heat_before` are meaningless for a
    post-call state and ignored. For a "no physics ran" probe, callers pass
    the SAME `PackedAmpsState` as both `packed_before` and `amps_after`
    (see `TestRoundTrip` in test_packing.py) -- every RHOQ_t/CPtot_t/CVtot_t
    /RHOE_t component is then ~0 by construction (F4's own conversion
    notebook, SS2.1: `x(n+1)*DEN(n+1) == X(n)*DEN(n)` exactly when `x(n+1)`
    is literally `pack`'s own output of `X(n)`).

    `l_gaxis_version=1` only (F4 SS2.4 "-- ORIGINAL", including its
    `l_axis_limit` `iag_q`/`icg_q` clip-overwrite); 2/3 raise
    `NotImplementedError` (M5 stub, see module docstring).
    """
    _require_gaxis_version_1(l_gaxis_version)
    _require_ncat_one(
        scale_before.liquid_ppv,
        scale_before.ice_ppv,
        scale_before.aerosol_ppv,
        amps_after.liquid,
        amps_after.ice,
        amps_after.aerosol,
    )

    dens = np.asarray(scale_before.dens, dtype=np.float64)
    dens_t = np.asarray(dens_t, dtype=np.float64)
    dens_new = dens + dens_t * dt  # DENS_NEW(k), F4 SS2.2

    moist_denv_after = get_thermo_prop(amps_after.thermo, ThermoProp.moist_denv)
    qv_after = get_thermo_prop(amps_after.thermo, ThermoProp.qvv)

    dqv = (qv_after * moist_denv_after - scale_before.qv * dens) / dt  # F4 SS2.2

    dql, d_liquid_ppv = _unpack_liquid(scale_before, amps_after, dens, moist_denv_after, dt)
    dqi, dqw, d_ice_ppv = _unpack_ice(
        scale_before, amps_after, dens, moist_denv_after, dt, l_axis_limit
    )
    d_aerosol_ppv = _unpack_aerosol(scale_before, amps_after, dens, moist_denv_after, dt)
    cptot_t, cvtot_t = _unpack_cptot_cvtot(
        scale_before, amps_after, dens_new, moist_denv_after, qv_after, dt, l_no_ice_heat
    )
    rhoe_t = _unpack_rhoe_t(
        packed_before, amps_after, moist_denv_after, qv_after, dt, l_no_ice_heat
    )

    return ScaleTendencies(
        dqv=dqv,
        dql=dql,
        dqi=dqi,
        dqw=dqw,
        d_liquid_ppv=d_liquid_ppv,
        d_ice_ppv=d_ice_ppv,
        d_aerosol_ppv=d_aerosol_ppv,
        cptot_t=cptot_t,
        cvtot_t=cvtot_t,
        rhoe_t=rhoe_t,
    )


# ---------------------------------------------------------------------------
# moistthermo_mask -- F4 SS3.2 (moistthermo2_scale, cloudy-mask + thil/qtp).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoistThermoMaskResult:
    """`moistthermo_mask`'s output, F4 SS3.2."""

    liquid: LiquidState  # input `liquid` with sub-RRLMTB bins zeroed (all props)
    ice: IceState  # input `ice` with sub-RILMTB bins zeroed (all props)
    qc: np.ndarray  # cloud liquid mixing ratio (bins below `ibr_st`), (npoints,)
    qr: np.ndarray  # rain liquid + ice-melt-water mixing ratio, (npoints,)
    qi: np.ndarray  # ice mixing ratio (aerosol + melt water subtracted), (npoints,)
    cnr: np.ndarray  # accumulated liquid number conc. of qualifying bins, (npoints,)
    micptr: np.ndarray  # cloudy mask, 0/1 int, (npoints,)
    qtp: np.ndarray  # total water content (qv + qualifying liquid/ice), (npoints,)
    thp: np.ndarray  # theta_il, (npoints,)


def moistthermo_mask(
    liquid: LiquidState,
    ice: IceState,
    qv: np.ndarray,
    th: np.ndarray,
    t: np.ndarray,
    *,
    nbhzcl: int,
) -> MoistThermoMaskResult:
    """`moistthermo2_scale`'s qc/qr/qi bin partition + cloudy mask (`micptr`)
    + `qtp`/`thp` diagnosis, F4 SS3.2 (lines 1053-1117). Does NOT implement
    SS3.3's iterative T-refinement/supersaturation-trigger machinery --
    out of this function's brief-scoped purpose (see module docstring).
    """
    _require_ncat_one(liquid, ice)

    qv = np.asarray(qv, dtype=np.float64)
    th = np.asarray(th, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    npoints = qv.shape[0]

    lp = index_maps.LiquidPPV
    ip = index_maps.IcePPV
    nbr = liquid.nbins
    nbi = ice.nbins
    ibr_st = nbhzcl + 1  # F4 SS3.2: `ibr_st=nbhzcl+1` (1-based Fortran)

    liquid_in = liquid.values  # read-only source (F4 accumulates qtotal BEFORE zeroing)
    liquid_out = liquid.values.copy()
    qc = np.zeros(npoints, dtype=np.float64)
    qr = np.zeros(npoints, dtype=np.float64)
    qi_out = np.zeros(npoints, dtype=np.float64)
    cnr = np.zeros(npoints, dtype=np.float64)
    micptr = np.zeros(npoints, dtype=np.int64)
    qtotal = np.zeros(npoints, dtype=np.float64)

    for ibr_0based in range(nbr):
        rmt = liquid_in[lp.rmt_q.py_idx, ibr_0based, 0, :]
        rmat = liquid_in[lp.rmat_q.py_idx, ibr_0based, 0, :]
        rcon = liquid_in[lp.rcon_q.py_idx, ibr_0based, 0, :]
        qtotal = qtotal + (rmt - rmat)
        passed = rmt >= RRLMTB
        contrib = np.maximum(0.0, rmt - rmat)
        is_cloud_bin = ibr_0based < ibr_st - 1  # Fortran `1..ibr_st-1` (1-based)
        if is_cloud_bin:
            qc = qc + np.where(passed, contrib, 0.0)
        else:
            qr = qr + np.where(passed, contrib, 0.0)
        micptr = np.where(passed, 1, micptr)
        cnr = cnr + np.where(passed, rcon, 0.0)
        liquid_out[:, ibr_0based, 0, :] = np.where(
            passed[None, :], liquid_out[:, ibr_0based, 0, :], 0.0
        )

    ice_in = ice.values
    ice_out = ice.values.copy()
    for ibi_0based in range(nbi):
        imt = ice_in[ip.imt_q.py_idx, ibi_0based, 0, :]
        imat = ice_in[ip.imat_q.py_idx, ibi_0based, 0, :]
        imw = ice_in[ip.imw_q.py_idx, ibi_0based, 0, :]
        qtotal = qtotal + (imt - imat)
        passed = imt >= RILMTB
        qi_contrib = np.maximum(0.0, imt - imat - imw)
        qr_contrib = np.maximum(0.0, imw)
        qi_out = qi_out + np.where(passed, qi_contrib, 0.0)
        qr = qr + np.where(passed, qr_contrib, 0.0)
        micptr = np.where(passed, 1, micptr)
        ice_out[:, ibi_0based, 0, :] = np.where(passed[None, :], ice_out[:, ibi_0based, 0, :], 0.0)

    qtotal = qtotal + qv
    qtp = qtotal

    aklv = SCALE_LHV0 / SCALE_CPDRY
    akiv = SCALE_LHS0 / SCALE_CPDRY
    thp = th / (1.0 + (aklv * (qr + qc) + akiv * qi_out) / np.maximum(t, 253.0))

    return MoistThermoMaskResult(
        liquid=LiquidState(values=liquid_out),
        ice=IceState(values=ice_out),
        qc=qc,
        qr=qr,
        qi=qi_out,
        cnr=cnr,
        micptr=micptr,
        qtp=qtp,
        thp=thp,
    )

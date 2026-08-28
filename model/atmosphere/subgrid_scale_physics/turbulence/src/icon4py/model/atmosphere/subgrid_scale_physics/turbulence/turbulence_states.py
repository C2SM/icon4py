# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""State containers of the NWP 1D turbulence granule (Raschendorfer scheme).

The inventory is the union of the dummy-argument lists of 'turbdiff' (turb_diffusion.f90:279),
'turbtran' (turb_transfer.f90:224) and 'vertdiff' (turb_vertdiff.f90:114), restricted to what
the ICON interfaces actually pass at 'mo_nwp_turbdiff_interface.f90:576' (turbdiff) and ':672'
(vertdiff) and at 'mo_nwp_turbtrans_interface.f90:555' and ':884' (the untiled and the tiled
turbtran call). The containers group by role, not by subroutine: a field that 'turbdiff' takes
as grid-mean and 'turbtran' as per-tile appears once in each of the two states that role
belongs to.

Arguments the interfaces never pass are left out. They are optional in the Fortran and
'PRESENT()' is false at every call site, so the branches behind them are dead (port spec 4.1)
and carrying them in the granule interface would suggest a capability that has no reference
data behind it:

* turbdiff (11 of its 79 dummy arguments): 'c_big', 'c_sml', 'r_air' -- the vertically
  resolved canopy, also switched off by the 'lporous = .FALSE.' parameter; 'tkhm', 'tkhh' --
  3D turbulence, hardcoded '.FALSE.' at mo_nwp_turbdiff_interface.f90:584; 'tket_sso',
  'tket_nstc', 'tket_buoy', 'tket_fshr', 'tket_gshr' -- TKE budget diagnostics ('tket_conv'
  and 'tket_hshr' are passed and are carried below); 'tketadv' -- TKE advection, so
  'lpres_avt = F'.
* turbtran (4 of 68): 'hdef2', 'dwdx', 'dwdy' -- the additional-shear machinery is fed to
  turbdiff only, never to turbtran; 'edr' -- the surface eddy dissipation rate output.
* vertdiff (5 of 50): 'dp0' -- optional here and not given, although turbdiff, where it is not
  optional, does receive it; 'r_air' -- as above; 'qv_conv' -- the interface marks the omission
  explicitly ("qv_conv: missing"); 'umfl_s', 'vmfl_s' -- turbtran writes them, vertdiff is not
  asked for the effective implicit values.

Two further arguments are deliberately not fields of any container. 'zvari(:,:,0:ndim)' is
granule-internal scratch that carries conserved variables, then gradients, then fluxes from
turbdiff to vertdiff; port spec 9.2 requires it to be split into named fields, so it never
appears as one array. 'ptr(:)' is a 'modvar' array of pointers rather than a field: its '%av'
and '%at' components become the 'tracers' and 'ddt_tracers' tuples below, while its '%kstart'
component is a scalar and belongs to the granule configuration.

Precision: 'turb_diffusion.f90:612' is the only 'REAL(KIND=vp)' declaration in all four scheme
files, covering 'hdef2', 'hdiv', 'dwdx' and 'dwdy', which come from the mixed-precision dycore
diffusion (mo_nh_diffusion.f90:847,:850,:1182,:1187). Exactly those four are typed 'vpfloat'
here; everything else is 'wpfloat'. The default build is double, where 'vpfloat is wpfloat',
so this costs nothing today and stays correct if mixed precision is ever enabled.

ADR-0001: a physics component returns tendencies and never mutates its input state. The
Fortran 'turbdiff' and 'vertdiff' update 'u', 'v', 't', 'qv' and 'qc' in place whenever the
optional '*_tens' arguments are absent; the ICON interfaces always pass them, and the granule
only ever writes 'TurbulenceTendencyState'. All containers are frozen.
"""

from __future__ import annotations

import dataclasses
from typing import Final, TypeAlias

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


__all__ = [
    "LAKE_TILE",
    "NUM_LAND_TILES",
    "NUM_TILES",
    "NUM_WATER_TILES",
    "OPEN_SEA_TILE",
    "SEA_ICE_TILE",
    "TileField",
    "TurbulenceDiagnosticState",
    "TurbulenceInputState",
    "TurbulenceMetricState",
    "TurbulenceSurfaceState",
    "TurbulenceTendencyState",
    "TurbulenceTileState",
]


#: Land sub-tiles: three land-use classes times {snow-free, snow-covered}. All six run
#: identical physics (port spec 7.2).
NUM_LAND_TILES: Final[int] = 6

#: Water tiles, in this order: open sea, lake, sea-ice. Each is special-cased.
NUM_WATER_TILES: Final[int] = 3

#: 'ntiles_total + ntiles_water' for every tiled NWP configuration. Not a free runtime count:
#: 'mo_lnd_nwp_config.f90:236-259' derives it from 'ntiles_lnd = 3' and 'lsnowtile = .TRUE.',
#: and ':227' forces 'lsnowtile = .FALSE.' when 'ntiles_lnd == 1'. So a run has either these
#: nine tiles or the degenerate single-tile shape with no water tiles at all.
NUM_TILES: Final[int] = NUM_LAND_TILES + NUM_WATER_TILES

#: Index of the open-sea tile within a per-tile tuple ('isub_water' in ICON, 1-based there).
OPEN_SEA_TILE: Final[int] = NUM_LAND_TILES

#: Index of the lake tile ('isub_lake').
LAKE_TILE: Final[int] = NUM_LAND_TILES + 1

#: Index of the sea-ice tile ('isub_seaice').
SEA_ICE_TILE: Final[int] = NUM_LAND_TILES + 2

#: A per-tile quantity: one dense cell field per surface tile, in ICON's tile order, holding
#: the same values as the third index of ICON's '_t' arrays.
#:
#: The tile axis is a Python tuple, not a field dimension. Port spec 7.3 rules out a 'TileDim'
#: (every non-horizontal middle axis in icon4py is a LOCAL connectivity dimension, and a plain
#: axis of runtime extent has no precedent) and equally rules out a Python-level loop over
#: tiles at call time. A tuple is neither: it is unrolled when the field operator is traced, so
#: the six land sub-tiles become six calls of one shared field operator inside a single fused
#: program, and the three water tiles are separate named entries selected statically.
TileField: TypeAlias = tuple[fa.CellField[ta.wpfloat], ...]


@dataclasses.dataclass(frozen=True)
class TurbulenceMetricState:
    """Vertical grid geometry and the static horizontal masks the scheme is given."""

    #: 'hhl' -- height of the model half levels, on half levels [m]. ICON passes
    #: 'p_metrics%z_ifc'; turbtran receives only its lowest three levels.
    hhl: fa.CellKField[ta.wpfloat]
    #: 'dp0' -- pressure thickness of a layer, on full levels [Pa]. ICON passes
    #: 'p_diag%dpres_mc'. Only turbdiff gets it; vertdiff declares it optional and is not
    #: given it.
    dp0: fa.CellKField[ta.wpfloat]
    #: 'l_hori' -- horizontal grid spacing [m]. Declared as a field, but ICON fills every entry
    #: with the single scalar 'phy_params%mean_charlen' (mo_nwp_turbdiff_interface.f90:535).
    l_hori: fa.CellField[ta.wpfloat]
    #: 'trop_mask' -- 1 within the tropics, 0 in the extra-tropics; scales the vertical
    #: smoothing of the TKE forcing terms. ICON passes 'prm_diag%tropics_mask'.
    trop_mask: fa.CellField[ta.wpfloat]
    #: 'innertrop_mask' -- as 'trop_mask', restricted to the inner tropics. ICON passes
    #: 'prm_diag%innertropics_mask'.
    innertrop_mask: fa.CellField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class TurbulenceInputState:
    """Atmospheric column input and the external forcings, read-only for the granule."""

    #: 'u' -- zonal wind at mass positions, on full levels [m/s].
    u: fa.CellKField[ta.wpfloat]
    #: 'v' -- meridional wind at mass positions, on full levels [m/s].
    v: fa.CellKField[ta.wpfloat]
    #: 'w' -- vertical wind, on half levels [m/s]. turbdiff only; not a turbtran argument.
    w: fa.CellKField[ta.wpfloat]
    #: 't' -- air temperature, on full levels [K].
    t: fa.CellKField[ta.wpfloat]
    #: 'qv' -- specific water vapour content, on full levels [kg/kg].
    qv: fa.CellKField[ta.wpfloat]
    #: 'qc' -- specific cloud water content, on full levels [kg/kg].
    qc: fa.CellKField[ta.wpfloat]
    #: 'prs' -- air pressure, on full levels [Pa].
    prs: fa.CellKField[ta.wpfloat]
    #: 'rhoh' -- total air density, on full levels [kg/m3]. The half-level density 'rhon' is
    #: computed by turbdiff and lives in `TurbulenceDiagnosticState`.
    rhoh: fa.CellKField[ta.wpfloat]
    #: 'epr' -- Exner pressure, on full levels [1].
    epr: fa.CellKField[ta.wpfloat]
    #: 'tke' -- the prognostic turbulent velocity q = sqrt(2 * TKE), on half levels [m/s].
    #: Not the TKE density. ICON runs the scheme with 'ntim = 1', so the Fortran time-level
    #: index of 'tke(:,:,ntim)' collapses and no time dimension is carried here.
    tke: fa.CellKField[ta.wpfloat]
    #: 'ptr(:)%av' -- the passive tracers vertdiff diffuses, on full levels, one field per
    #: tracer. Which tracers these are is a runtime choice of the interface (qi, qs and the
    #: two-moment number densities, plus any ART tracers), hence 'ndtr' entries rather than
    #: named fields.
    tracers: tuple[fa.CellKField[ta.wpfloat], ...]
    #: 'ut_sso' -- zonal wind tendency of the SSO scheme, on full levels [m/s2]. An external
    #: forcing of the TKE budget, not something the granule computes.
    ut_sso: fa.CellKField[ta.wpfloat]
    #: 'vt_sso' -- meridional wind tendency of the SSO scheme, on full levels [m/s2].
    vt_sso: fa.CellKField[ta.wpfloat]
    #: 'tket_conv' -- TKE tendency from convective buoyancy, on half levels [m2/s3]. ICON
    #: passes 'prm_nwp_tend%ddt_tke_pconv'. The only TKE budget term that is an input; the
    #: other 'tket_*' arguments are never passed.
    tket_conv: fa.CellKField[ta.wpfloat]
    #: 'hdef2' -- squared horizontal deformation, on half levels [1/s2]. From the dycore
    #: diffusion ('p_diag%hdef_ic', mo_nh_diffusion.f90:847), hence 'vpfloat'.
    hdef2: fa.CellKField[ta.vpfloat]
    #: 'hdiv' -- horizontal divergence, on half levels [1/s]. From the dycore diffusion
    #: ('p_diag%div_ic', mo_nh_diffusion.f90:850), hence 'vpfloat'.
    hdiv: fa.CellKField[ta.vpfloat]
    #: 'dwdx' -- zonal derivative of the vertical wind, on half levels [1/s]. From the dycore
    #: diffusion ('p_diag%dwdx', mo_nh_diffusion.f90:1182), hence 'vpfloat'.
    dwdx: fa.CellKField[ta.vpfloat]
    #: 'dwdy' -- meridional derivative of the vertical wind, on half levels [1/s]. From the
    #: dycore diffusion ('p_diag%dwdy', mo_nh_diffusion.f90:1187), hence 'vpfloat'.
    dwdy: fa.CellKField[ta.vpfloat]


@dataclasses.dataclass(frozen=True)
class TurbulenceSurfaceState:
    """Grid-mean surface state and external-parameter fields.

    These are the surface quantities the granule reads as they are. The ones ICON keeps per
    tile and iterates on across time steps are in `TurbulenceTileState` instead.
    """

    #: 't_g' -- weighted surface temperature [K]. ICON passes 'lnd_prog%t_g' to turbdiff and
    #: vertdiff; turbtran gets the per-tile 't_g_t'.
    t_g: fa.CellField[ta.wpfloat]
    #: 'qv_s' -- specific water vapour content at the surface [kg/kg]. Grid-mean
    #: 'lnd_diag%qv_s'; turbtran gets the per-tile 'qv_s_t'.
    qv_s: fa.CellField[ta.wpfloat]
    #: 'ps' -- surface pressure [Pa]. ICON passes 'p_diag%pres_sfc'.
    ps: fa.CellField[ta.wpfloat]
    #: 'fr_land' -- land portion of the grid point area [1].
    fr_land: fa.CellField[ta.wpfloat]
    #: 'l_lake' -- the surface is a lake. Derived in the turbtrans interface, not an external
    #: parameter.
    l_lake: fa.CellField[bool]
    #: 'l_sice' -- the surface is sea or lake ice. Frozen land points are excluded.
    l_sice: fa.CellField[bool]
    #: 'l_pat' -- effective length scale of the thermal inhomogeneities of the surface [m],
    #: which scales the near-surface circulation acceleration.
    l_pat: fa.CellField[ta.wpfloat]
    #: 'urb_isa' -- urban impervious surface area [1]. ICON stores it per tile
    #: ('ext_data%atm%urb_isa_t') and turbtran always receives the value of the tile being
    #: computed; it is carried grid-mean here because it is a static external parameter that
    #: the granule never updates.
    urb_isa: fa.CellField[ta.wpfloat]
    #: 'rlamh_fac' -- scaling factor for the laminar heat resistance 'rlam_heat' [1]. Also a
    #: per-tile array in ICON ('prm_diag%rlamh_fac_t'), and read-only for the same reason.
    rlamh_fac: fa.CellField[ta.wpfloat]
    #: 'z0_waves' -- roughness length supplied by the wave model [m]. Only used when the sea
    #: tile is wave-coupled ('igz0inp == 2'); zero otherwise.
    z0_waves: fa.CellField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class TurbulenceDiagnosticState:
    """Diagnostic fields of the turbulence model.

    Grid-mean quantities that the scheme both reads and writes and that ICON carries from one
    time step to the next, plus the pure outputs (near-surface diagnostics, surface fluxes).
    """

    #: 'gz0' -- roughness length times gravity of the vertically unresolved roughness layer
    #: [m2/s2]. turbtran updates it per tile and the interface aggregates it back into the
    #: grid-mean 'prm_diag%gz0' that turbdiff reads.
    gz0: fa.CellField[ta.wpfloat]
    #: 'tcm' -- turbulent transfer coefficient for momentum [1]. Output of turbtran.
    tcm: fa.CellField[ta.wpfloat]
    #: 'tch' -- turbulent transfer coefficient for heat and moisture [1].
    tch: fa.CellField[ta.wpfloat]
    #: 'tvm' -- turbulent transfer velocity for momentum [m/s]. Together with 'tvh' this
    #: supersedes 'tcm'/'tch' inside the scheme.
    tvm: fa.CellField[ta.wpfloat]
    #: 'tvh' -- turbulent transfer velocity for heat and moisture [m/s].
    tvh: fa.CellField[ta.wpfloat]
    #: 'tfm' -- Prandtl-layer fraction of the total transfer-layer resistance for momentum [1]
    #: on output of turbtran; on output of turbdiff the factor that removes the pure drag
    #: contribution of 'tkmmin'. The meaning of the slot differs between the two subroutines.
    tfm: fa.CellField[ta.wpfloat]
    #: 'tfh' -- as 'tfm' for scalars on output of turbtran; on output of turbdiff, the spurious
    #: shear forcing implied by clipping the diffusion coefficients at their lower limits
    #: (LLDCs, "Lower Limits of Diffusion-Coefficients", mo_turbdiff_config.f90:170-174) at the
    #: "P" level [1/s2]. It is an artefact of the tkhmin/tkmmin floor, not a physical shear mode
    #: (turb_diffusion.f90:1979-1980).
    tfh: fa.CellField[ta.wpfloat]
    #: 'tfv' -- additional shear forcing by non-turbulent subgrid circulations (NTCs) at the "P"
    #: level [1/s2]: SSO wakes, separated horizontal shear, convective and near-surface thermal
    #: circulations together. Computed as the total mechanical forcing minus the pure-mean-shear
    #: forcing, 'frm - ftm' (turb_diffusion.f90:1677), i.e. the whole scale-interaction residual;
    #: near-surface thermals alone have their own slot, 'tket_nstc'.
    tfv: fa.CellField[ta.wpfloat]
    #: 'tkred_sfc' -- reduction factor for the minimum momentum diffusion coefficient near the
    #: surface [1].
    tkred_sfc: fa.CellField[ta.wpfloat]
    #: 'tkred_sfc_h' -- the same for the scalar diffusion coefficient [1].
    tkred_sfc_h: fa.CellField[ta.wpfloat]
    #: 'shfl_s' -- sensible heat flux at the surface [W/m2], positive downward. Written by
    #: turbtran and, when 'lsflcnd' holds, used by vertdiff as the lower boundary condition.
    shfl_s: fa.CellField[ta.wpfloat]
    #: 'qvfl_s' -- water vapour flux at the surface [kg/(m2 s)], positive downward. ICON calls
    #: it 'prm_diag%qhfl_s'.
    qvfl_s: fa.CellField[ta.wpfloat]
    #: 'umfl_s' -- zonal momentum flux at the surface [N/m2], positive downward. Written by
    #: turbtran; vertdiff is not asked for the effective implicit value.
    umfl_s: fa.CellField[ta.wpfloat]
    #: 'vmfl_s' -- meridional momentum flux at the surface [N/m2], positive downward.
    vmfl_s: fa.CellField[ta.wpfloat]
    #: 't_2m' -- temperature at 2 m [K]. The 2 m and 10 m diagnostics themselves are live; only
    #: the level search behind them is dead (port spec 4.1).
    t_2m: fa.CellField[ta.wpfloat]
    #: 'qv_2m' -- specific water vapour content at 2 m [kg/kg].
    qv_2m: fa.CellField[ta.wpfloat]
    #: 'td_2m' -- dew point at 2 m [K].
    td_2m: fa.CellField[ta.wpfloat]
    #: 'rh_2m' -- relative humidity at 2 m [%].
    rh_2m: fa.CellField[ta.wpfloat]
    #: 'u_10m' -- zonal wind at 10 m [m/s].
    u_10m: fa.CellField[ta.wpfloat]
    #: 'v_10m' -- meridional wind at 10 m [m/s].
    v_10m: fa.CellField[ta.wpfloat]
    #: 'tkvm' -- turbulent diffusion coefficient for momentum, on half levels [m2/s].
    tkvm: fa.CellKField[ta.wpfloat]
    #: 'tkvh' -- turbulent diffusion coefficient for heat and other scalars, on half levels
    #: [m2/s].
    tkvh: fa.CellKField[ta.wpfloat]
    #: 'tprn' -- turbulent Prandtl number, on half levels [1].
    tprn: fa.CellKField[ta.wpfloat]
    #: 'rcld' -- standard deviation of the local supersaturation, at main levels including the
    #: lower boundary, so 'nlev + 1' entries [1]. Inside the scheme the slot is also used for
    #: the cloud cover, first at main and later at half levels.
    rcld: fa.CellKField[ta.wpfloat]
    #: 'rhon' -- total air density on half levels [kg/m3]. Output of turbdiff, input of
    #: vertdiff; ICON keeps it in the interface-local 'zrhon'.
    rhon: fa.CellKField[ta.wpfloat]
    #: 'edr' -- eddy dissipation rate of TKE, on half levels [m2/s3]. A pointer argument that
    #: ICON leaves disassociated unless 'ldiagnose_tke' is set
    #: (mo_nwp_turbdiff_interface.f90:309).
    edr: fa.CellKField[ta.wpfloat]
    #: 'tur_len_scale' -- turbulent length scale, on half levels [m]. Output-only, and
    #: disassociated under the same condition as 'edr'.
    tur_len_scale: fa.CellKField[ta.wpfloat]
    #: 'tke(:,:,ntur)' -- the turbulent velocity q = sqrt(2 * TKE) the scheme produces, on half
    #: levels [m/s]. The counterpart of `TurbulenceInputState.tke`, which is 'tke(:,:,nvor)',
    #: the level the iteration starts from.
    #:
    #: ICON runs with 'ntim = 1', so 'nvor == ntur' and the Fortran updates one array in place.
    #: The port cannot: ADR-0001 forbids a physics component writing into its input state, and
    #: the port spec lists the TKE time levels among the things that "need mapping onto discrete
    #: fields" (9.2). So the two levels are two fields here and the granule reads one and writes
    #: the other. Only the surface half level is neither: 'turbtran' owns it and 'turbdiff'
    #: leaves it alone, so the granule copies it across before section 3) runs.
    updated_tke: fa.CellKField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class TurbulenceTendencyState:
    """Tendencies the granule produces.

    Kept apart from `TurbulenceInputState` on purpose: the Fortran updates 'u', 'v', 't', 'qv'
    and 'qc' in place when the optional '*_tens' arguments are absent, which ADR-0001 forbids
    for an icon4py physics component.
    """

    #: 'u_tens' -- zonal wind tendency, on full levels [m/s2]. ICON passes
    #: 'prm_nwp_tend%ddt_u_turb'.
    ddt_u: fa.CellKField[ta.wpfloat]
    #: 'v_tens' -- meridional wind tendency, on full levels [m/s2].
    ddt_v: fa.CellKField[ta.wpfloat]
    #: 't_tens' -- temperature tendency, on full levels [K/s]. ICON passes
    #: 'prm_nwp_tend%ddt_temp_turb'.
    ddt_t: fa.CellKField[ta.wpfloat]
    #: 'qv_tens' -- specific water vapour tendency, on full levels [1/s]. Written by vertdiff
    #: only.
    ddt_qv: fa.CellKField[ta.wpfloat]
    #: 'qc_tens' -- specific cloud water tendency, on full levels [1/s]. Written by vertdiff
    #: only.
    ddt_qc: fa.CellKField[ta.wpfloat]
    #: 'tketens' -- diffusion tendency of q = sqrt(2 * TKE), on half levels [m/s2]. The one
    #: tendency argument of turbdiff that is not optional.
    ddt_tke: fa.CellKField[ta.wpfloat]
    #: 'ptr(:)%at' -- tendencies of the diffused passive tracers, on full levels, aligned
    #: entry by entry with `TurbulenceInputState.tracers`.
    ddt_tracers: tuple[fa.CellKField[ta.wpfloat], ...]
    #: 'tket_hshr' -- TKE tendency from separated horizontal shear, on half levels [m2/s3].
    #: A diagnostic output ('prm_nwp_tend%ddt_tke_hsh'), not fed back into the budget here.
    tket_hshr: fa.CellKField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class TurbulenceTileState:
    """Per-tile surface state of turbtran.

    turbtran has no tile dimension of its own: ICON calls it once per tile on gathered index
    lists (mo_nwp_turbtrans_interface.f90:319, :677) and aggregates the results afterwards.
    These are the quantities that are genuinely per-tile *state* -- iterated on from one call
    of turbtran to the next -- rather than per-tile outputs, which the interface aggregates and
    which therefore appear grid-mean in `TurbulenceDiagnosticState`.

    Every field is a tuple of one cell field per tile, in ICON's tile order: `NUM_LAND_TILES`
    land sub-tiles running identical physics, then open sea, lake and sea-ice. See `TileField`
    for why the tile axis is a tuple and not a dimension.
    """

    #: 'gz0_t' -- roughness length times gravity, per tile [m2/s2]. turbtran updates it and it
    #: is read back on the next call ('prm_diag%gz0_t').
    gz0_t: TileField
    #: 'sai_t' -- surface area index, per tile [1]. For land tiles the interface blends in the
    #: snow-cover contribution before the call ('ext_data%atm%sai_t').
    sai_t: TileField
    #: 't_g_t' -- surface temperature, per tile [K] ('lnd_prog%t_g_t').
    t_g_t: TileField
    #: 'qv_s_t' -- specific water vapour content at the surface, per tile [kg/kg]
    #: ('lnd_diag%qv_s_t').
    qv_s_t: TileField
    #: 'frac_t' -- area fraction of the tile [1] ('ext_data%atm%frac_t'). The weight of the
    #: aggregation back to grid-mean, and the guard that keeps values on inactive tiles out of
    #: the sum (port spec 7.2).
    frac_t: TileField
    #: 'tvs_s_t' -- turbulent velocity q = sqrt(2 * TKE) at the surface level, per tile [m/s]
    #: ('prm_diag%tvs_s_t').
    tvs_s_t: TileField
    #: 'tkvm_s_t' -- momentum diffusion coefficient at the surface level, per tile [m2/s].
    tkvm_s_t: TileField
    #: 'tkvh_s_t' -- scalar diffusion coefficient at the surface level, per tile [m2/s].
    tkvh_s_t: TileField
    #: 'rcld_s_t' -- standard deviation of the local supersaturation at the surface level, per
    #: tile [1].
    rcld_s_t: TileField
    #: 'tkr_t' -- reciprocal dimensionless diffusion coefficient at the top of the roughness
    #: layer, u* / (q * Sm)_0, per tile [1]. Needed across time steps when
    #: 'imode_trancnf >= 4'.
    tkr_t: TileField

    def __post_init__(self) -> None:
        counts = {field.name: len(getattr(self, field.name)) for field in dataclasses.fields(self)}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Ragged tile axis: every per-tile field must agree, got {counts}.")
        if self.num_tiles not in (1, NUM_TILES):
            raise ValueError(
                f"Invalid number of tiles: expected 1 or {NUM_TILES}, got {self.num_tiles}."
            )

    @property
    def num_tiles(self) -> int:
        """Number of tiles carried, either `NUM_TILES` or 1 for the untiled configuration."""
        return len(self.frac_t)

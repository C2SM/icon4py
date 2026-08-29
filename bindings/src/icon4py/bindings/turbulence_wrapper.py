# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""
Wrapper module for the NWP 1D turbulence granule (Raschendorfer scheme).

Module contains a turbulence_init and a turbulence_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types

'turbulence_run' is one call of 'Turbulence.run', which is 'turbdiff' followed by 'vertdiff'.
That pair is the unit 'mo_nwp_turbdiff_interface.f90' substitutes: it calls 'turbdiff' at :576
and 'vertdiff' at :672 with nothing between them but a timer, and the two calls share 'rhon',
'zvari', 'tkvm', 'tkvh', 'tvm' and 'tvh'. 'turbtran' is NOT part of this -- ICON calls it from
'mo_nwp_turbtrans_interface.f90', once per surface tile -- and it is a later phase of the port.

WHAT CROSSES THE C BOUNDARY, AND WHY
------------------------------------
Flat at the C boundary, structured inside (port spec D7): every argument is a scalar or a plain
array, and the state containers of 'turbulence_states.py' are assembled from them on this side.

* EVERY member of 'TurbulenceConfig' is an argument of 'turbulence_init', in declaration order.
  The container is exactly 'turbdiff_nml' plus those members of 't_turbdiff_config' that select
  a formulation, so the Fortran caller fills the whole list from its own 'tdc' state field by
  field. Most of them the implementation only validates -- 'FROZEN_SWITCHES' refuses the
  formulations that were not ported -- but the interface is the contract and has to be
  expressible from Fortran (port spec D5/D6). A switch absent here is a switch ICON could set
  without the granule ever noticing.

* Every field 'Turbulence.run' reads or writes is an argument, under the dummy-argument name of
  the ICON call site, so the Fortran side is a transcription and not a translation.

* Fields the state containers declare but 'run' never touches are NOT arguments. They are
  allocated once here and filled with NaN, so a granule that starts reading one poisons its
  output instead of silently consuming a plausible zero. '_allocate_the_unused_state_fields'
  lists each with the ICON argument it stands for. The turbtran-only members are the bulk of
  them; 'w', 'tket_conv', 'tfv', 'tprn', 'edr' and 'tur_len_scale' are the ones ICON does pass
  to 'turbdiff' and the ported scheme provably does not read or write.

THE TKE ARRAY IS ONE ARRAY IN ICON AND TWO FIELDS IN THE GRANULE.
ICON runs with 'ntim = 1', passes a single 'tke=z_tvs(:,:)' as INTENT(INOUT), and reads it back
at ':812,:825' to update 'p_prog_rcf%tke'. The granule cannot do that: ADR-0001 forbids a
physics component writing into its input state, so 'TurbulenceInputState.tke' and
'TurbulenceDiagnosticState.updated_tke' are two fields. 'turbulence_run' takes the one array
ICON has, seeds the granule's output buffer from it and copies the result back. The seed is what
keeps the columns outside the granule's horizontal window -- 'turbdiff' computes 'ivstart..ivend'
only -- at the value ICON gave them.

'dp0' IS PASSED TO 'turbulence_init' AND ALIASES ICON'S ARRAY -- WHICH IS WHY IT WORKS.
It is a member of 'TurbulenceMetricState', which the granule takes once, and two of its stencils
bind it as a 'constant_args' field at setup time. ICON's 'p_diag%dpres_mc' is not constant: it is
recomputed every step. The binding is BY IDENTITY and not by value -- 'setup_program' inlines only
scalars -- and py2fgen wraps the caller's memory instead of copying it, so the bound field reads
ICON's current values at every call.

That was written down as a suspected latent bug. IT IS NOT ONE, and the evidence is L3, not an
argument: 'ICON4PY_MODE_VERIFY' (job 834636, byte-identical in 834659) ran six timesteps with the
granule and the Fortran turbulence side by side, each step restarting from the Fortran state. A
'dp0' that had gone stale would agree at step 1, where the bound array still holds what ICON had
just written, and diverge from step 2 onwards; that asymmetry IS the test. Nothing diverged --
step-to-step ratios are non-monotone jitter in both directions, and the worst relative
disagreement across all 26 turbulence fields is 1.476e-06 on 't_tens', an absolute error of
1.1e-14 on a field of order 1.3e-02.

The precondition belongs to the caller and is not checked here: ICON must keep 'p_diag%dpres_mc'
in the same allocation for the life of the granule, and on GPU its device pointer must stay
stable. See the note in 'turbulence_init'.

NO TRACER REACHES THE GRANULE, AND NOTHING ON THIS SIDE WOULD SAY SO.
'turbulence_run' has no tracer argument. 'TurbulenceInputState.tracers' and
'TurbulenceTendencyState.ddt_tracers' are TUPLES of fields, and a tuple has no flat
representation at the C boundary: 'ndtr' is a runtime number while py2fgen renders a fixed
argument list. So this wrapper passes 'tracers=()' and 'ddt_tracers=()' unconditionally, and
'vertdiff' diffuses the five first-order variables and nothing else.

That is right only where 'ndtr = 0'. Switch on 'ldiff_qi' or 'ldiff_qs', two-moment or SBM
microphysics, ART or ComIn tracers, and ICON would hand the interface tracers this granule would
SILENTLY NOT DIFFUSE: no exception, no warning, output that looks entirely plausible and is
missing a physical process.

WHAT PREVENTS THAT TODAY LIVES IN A DIFFERENT REPOSITORY. 'check_supported_configuration' in
ICON's 'mo_icon4py_turbulence.f90' calls 'finish' when 'nturb_tracer_tot > 0'. It is the only
guard there is, it is not in this tree, and nobody reading this file can see it. If it is ever
removed -- or the granule is driven from anywhere else: the green line, a standalone driver, a
second wrapper -- this turns into a silent wrong answer with no failing test anywhere. The fix is
to give 'turbulence_run' the tracers, as a fixed maximum count with an active-count argument or
as a single '(ndtr, ncells, nlev)' array; refusing 'ndtr > 0' on this side would be second best
and still better than depending on a guard in another repository.

CONSIDERED AND DEFERRED
-----------------------
Three shapes of this interface were questioned while it was written and are deliberately left
alone. An independent scientific review is the next milestone, and churning the API immediately
before it would invalidate the reading it is about to get. Recorded here so a reviewer meets the
question instead of rediscovering it, and so that "nobody thought of it" is not the conclusion.

* PER-STAGE STATE CONTAINERS. The five containers of 'turbulence_states.py' are shaped by the
  ICON call site rather than by the stage, so they also carry turbtran's members -- which is why
  '_UNUSED_STATE_FIELDS' has 22 entries to NaN-fill here. Splitting them per stage would shorten
  this file. It would also renumber argument lists the reviewer is about to read, and those
  turbtran members are the specification of the next phase of the port, not dead weight.

* THE COMPUTED COLUMN WINDOW IS PRIVATE. The granule knows which columns it writes --
  '_start_cell' and '_end_cell', 'h_grid.Zone.NUDGING' to 'h_grid.Zone.LOCAL' -- and does not
  expose them. 'num_cells' is the right clamp for the two raw-array copies below, because they
  seed the whole 'num_cells' range first; it is the WRONG clamp for anything written back
  without such a seed, and using the one where the other was needed was a real defect that only
  L3 could see (integration design note D-I17). Two legitimate ranges, and this file can only
  name one of them. A public property would fix that. Adding public API is exactly what this
  pass is not doing.

* 'turbulence_init' TAKES 99 ARGUMENTS, 93 OF THEM CONFIGURATION. That is the point rather than
  an accident (port spec D5/D6): a switch missing from the list is a switch ICON can set without
  the granule ever noticing. No shorter flat form keeps that property, and
  'test_turbulence_init_builds_the_configuration_the_flat_arguments_describe' builds the call
  from 'dataclasses.fields', so a member this signature does not accept fails loudly with the
  name that moved.
"""

import dataclasses
import logging
import math

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

from icon4py.bindings import (
    common as wrapper_common,
    config as wrapper_config,
    grid_wrapper,
    icon4py_export,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence,
    turbulence_options as options,
    turbulence_states as states,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_backends
from icon4py.model.common.grid import base as grid_base
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TurbulenceGranule:
    """What 'turbulence_init' builds and 'turbulence_run' uses."""

    turbulence: turbulence.Turbulence
    #: 'tke(:,:,ntur)', the granule's TKE output. Not an argument of 'turbulence_run': see the
    #: module docstring on why ICON's one array becomes two fields here.
    updated_tke: gtx.Field
    #: 'p_patch%n_patch_cells', the width of every field allocated here. NOT the width of the
    #: fields ICON passes in: those are '(:,:,jb)' slices of '(nproma, nlev, nblks)' and are
    #: 'nproma' wide, which is strictly larger -- 'icon4py_init' requires
    #: 'nproma >= n_patch_edges' and edges always outnumber cells. Held on the granule so the
    #: two raw-array copies in 'turbulence_run' can say which of the two lengths they mean.
    num_cells: int
    #: The state-container members 'Turbulence.run' never touches, NaN-filled.
    unused: dict[str, gtx.Field]


granule: TurbulenceGranule | None = None


#: The members of the five state containers that 'Turbulence.run' neither reads nor writes,
#: with the ICON argument each stands for. Keeping them out of the wrapper's signature keeps
#: every argument of 'turbulence_run' load-bearing; NaN-filling them keeps the omission honest.
#:
#: name -> (shape, why it is not an argument)
_UNUSED_STATE_FIELDS: dict[str, tuple[str, str]] = {
    # -- passed by ICON to 'turbdiff', not read by the ported scheme
    "w": (
        "half",
        "'p_prog%w'. The vertical velocity enters the shear only through 'dwdx', "
        "'dwdy' and 'hdef2', which the dycore diffusion computes and which are arguments.",
    ),
    "tket_conv": (
        "half",
        "'prm_nwp_tend%ddt_tke_pconv'. Read only under 'ltkecon', frozen "
        "'.FALSE.' in 'FROZEN_SWITCHES'.",
    ),
    "tfv": (
        "surface",
        "'prm_diag%tfv'. Written only under 'rsur_sher > 0'; measured unchanged "
        "at every section savepoint of the capture.",
    ),
    "tprn": (
        "half",
        "'prm_diag%tprn'. Not written in this configuration; ICON allocates it "
        "degenerately when it is not diagnosed.",
    ),
    "edr": (
        "half",
        "'edr_ptr'. Disassociated unless 'ldiagnose_tke' (mo_nwp_turbdiff_interface.f90:309).",
    ),
    "tur_len_scale": ("half", "'len_scale_ptr'. Disassociated under the same condition."),
    # -- turbtran state, not an argument of either ICON call this wrapper replaces
    "fr_land": ("surface", "'ext_data%atm%fr_land'. turbtran only."),
    "l_lake": ("bool", "'l_lake'. turbtran only."),
    "l_sice": ("bool", "'l_sice'. turbtran only."),
    "urb_isa": ("surface", "'ext_data%atm%urb_isa_t'. turbtran only."),
    "rlamh_fac": ("surface", "'prm_diag%rlamh_fac_t'. turbtran only."),
    "z0_waves": ("surface", "'z0_waves'. turbtran only."),
    "tcm": (
        "surface",
        "'prm_diag%tcm'. Output of turbtran; superseded inside the scheme by "
        "'tvm', which is an argument.",
    ),
    "tch": ("surface", "'prm_diag%tch'. As 'tcm', superseded by 'tvh'."),
    "umfl_s": (
        "surface",
        "'prm_diag%umfl_s'. Written by turbtran; the ICON vertdiff call "
        "does not ask for the effective implicit value (:727-733).",
    ),
    "vmfl_s": ("surface", "'prm_diag%vmfl_s'. As 'umfl_s'."),
    "t_2m": ("surface", "'prm_diag%t_2m'. A turbtran diagnostic."),
    "qv_2m": ("surface", "'prm_diag%qv_2m'. A turbtran diagnostic."),
    "td_2m": ("surface", "'prm_diag%td_2m'. A turbtran diagnostic."),
    "rh_2m": ("surface", "'prm_diag%rh_2m'. A turbtran diagnostic."),
    "u_10m": ("surface", "'prm_diag%u_10m'. A turbtran diagnostic."),
    "v_10m": ("surface", "'prm_diag%v_10m'. A turbtran diagnostic."),
}


def _allocate_the_unused_state_fields(
    grid: grid_base.Grid, allocator: gtx_typing.Allocator | None
) -> dict[str, gtx.Field]:
    """Allocate one poisoned field per entry of '_UNUSED_STATE_FIELDS'.

    NaN rather than zero: a zero is a plausible value for every one of these quantities, so a
    granule that starts reading one would keep producing numbers. The two boolean masks cannot
    carry NaN and are allocated 'False', which is stated in '_UNUSED_STATE_FIELDS' as the one
    place this guard does not hold.
    """
    fields: dict[str, gtx.Field] = {}
    for name, (shape, _) in _UNUSED_STATE_FIELDS.items():
        if shape == "bool":
            fields[name] = data_alloc.zero_field(
                grid, dims.CellDim, dtype=bool, allocator=allocator
            )
            continue
        field = (
            data_alloc.zero_field(grid, dims.CellDim, allocator=allocator)
            if shape == "surface"
            else data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
            )
        )
        field.ndarray[...] = math.nan  # type: ignore[index]  # NDArrayObject Protocol
        fields[name] = field
    return fields


@icon4py_export.export
def turbulence_init(  # noqa: PLR0917 [too-many-positional-arguments]
    # -- 'TurbulenceMetricState': the static geometry and the horizontal masks.
    hhl: fa.CellKField[wpfloat],
    dp0: fa.CellKField[wpfloat],
    l_hori: fa.CellField[wpfloat],
    trop_mask: fa.CellField[wpfloat],
    innertrop_mask: fa.CellField[wpfloat],
    # -- 'TurbulenceConfig', member by member, in declaration order. Every one of them is
    #    'tdc%<name>' on the Fortran side.
    # -- 1. Numerical parameters (mo_turbdiff_config.f90:124-153)
    impl_s: gtx.float64,
    impl_t: gtx.float64,
    imode_tkvmini: gtx.int32,
    tkhmin: gtx.float64,
    tkmmin: gtx.float64,
    tkhmin_strat: gtx.float64,
    tkmmin_strat: gtx.float64,
    ditsmot: gtx.float64,
    imode_frcsmot: gtx.int32,
    frcsmot: gtx.float64,
    tkesmot: gtx.float64,
    frcsecu: gtx.float64,
    tkesecu: gtx.float64,
    stbsecu: gtx.float64,
    prfsecu: gtx.float64,
    epsi: gtx.float64,
    it_end: gtx.int32,
    # -- 2. Physical properties of the lower boundary (:155-190)
    rlam_heat: gtx.float64,
    rlam_mom: gtx.float64,
    rat_lam: gtx.float64,
    rat_sea: gtx.float64,
    rat_glac: gtx.float64,
    rat_can: gtx.float64,
    imode_nsf_wind: gtx.int32,
    rsur_sher: gtx.float64,
    imode_charpar: gtx.int32,
    alpha0: gtx.float64,
    alpha0_max: gtx.float64,
    alpha0_pert: gtx.float64,
    alpha1: gtx.float64,
    # -- 3. Stand-ins for external parameter fields (:192-207)
    c_lnd: gtx.float64,
    c_sea: gtx.float64,
    c_soil: gtx.float64,
    c_stm: gtx.float64,
    e_surf: gtx.float64,
    lconst_z0: bool,
    const_z0: gtx.float64,
    # -- 4. Stand-ins for dynamical fields (:209-213)
    z0m_dia: gtx.float64,
    z0_ice: gtx.float64,
    # -- 5. Turbulent diffusion parameters (:215-254)
    tur_len: gtx.float64,
    pat_len: gtx.float64,
    len_min: gtx.float64,
    imode_vel_min: gtx.int32,
    vel_min: gtx.float64,
    vel_max: gtx.float64,
    akt: gtx.float64,
    a_heat: gtx.float64,
    a_mom: gtx.float64,
    d_heat: gtx.float64,
    d_mom: gtx.float64,
    c_diff: gtx.float64,
    a_stab: gtx.float64,
    a_hshr: gtx.float64,
    clc_diag: gtx.float64,
    q_crit: gtx.float64,
    c_scld: gtx.float64,
    # -- 6. Switches (:260-284)
    ltkesso: bool,
    ltkecon: bool,
    ltkeshs: bool,
    ltkenst: bool,
    loutsso: bool,
    loutshs: bool,
    loutnst: bool,
    loutbms: bool,
    ltmpcor: bool,
    lcpfluc: bool,
    lexpcor: bool,
    lsflcnd: bool,
    lcirflx: bool,
    ldiff_qi: bool,
    ldiff_qs: bool,
    lfreeslip: bool,
    l3dturb: bool,
    # -- 7. Selectors (:290-389)
    imode_tran: gtx.int32,
    imode_turb: gtx.int32,
    icldm_tran: gtx.int32,
    icldm_turb: gtx.int32,
    itype_wcld: gtx.int32,
    itype_sher: gtx.int32,
    imode_stbcalc: gtx.int32,
    ilow_def_cond: gtx.int32,
    imode_pat_len: gtx.int32,
    imode_shshear: gtx.int32,
    imode_tkesso: gtx.int32,
    imode_snowsmot: gtx.int32,
    itype_2m_diag: gtx.int32,
    imode_stadlim: gtx.int32,
    imode_trancnf: gtx.int32,
    imode_lamdiff: gtx.int32,
    imode_tkemini: gtx.int32,
    imode_suradap: gtx.int32,
    imode_tkediff: gtx.int32,
    imode_adshear: gtx.int32,
    backend: gtx.int32,
) -> None:
    """Configure the turbulence granule and build its working set, once.

    'grid_init' must have run first: the granule needs the horizontal grid for its domain zones
    and the vertical grid for 'vct_a', from which it rebuilds ICON's implicit weight exactly as
    'mo_nwp_phy_init.f90:1541-1547' does.

    Args:
        hhl: 'p_metrics%z_ifc' -- half-level heights [m].
        dp0: 'p_diag%dpres_mc' -- layer pressure thickness [Pa]. THIS ONE IS NOT STATIC. It is
            a member of the granule's metric state, which is taken once, and two stencils bind
            it as a constant field at setup time; ICON recomputes it every step. The binding
            follows those updates because it is by identity and py2fgen wraps the caller's
            memory instead of copying it -- and that is not reasoning, it is what L3 measured
            over six timesteps (job 834636; see the module docstring). What ICON has to hold up
            its end of: the same allocation for the life of the granule, and a stable device
            pointer on GPU. If either ever stops being true, 'turbulence_init' has to be called
            again -- or 'dp0' has to move out of 'TurbulenceMetricState'.
        l_hori: 'l_hori' -- horizontal grid spacing [m], filled with 'phy_params%mean_charlen'.
        trop_mask: 'prm_diag%tropics_mask'.
        innertrop_mask: 'prm_diag%innertropics_mask'.
        backend: 'BackendIntEnum' selecting the GT4Py backend.
    """
    if grid_wrapper.grid_state is None:
        raise Exception(
            "Need to initialise grid using 'grid_init' before running 'turbulence_init'."
        )

    xp = hhl.array_ns  # type: ignore[attr-defined]  # to be fixed in gt4py
    on_gpu = xp != np  # TODO(havogt): expose `on_gpu` from py2fgen
    actual_backend = wrapper_common.select_backend(
        wrapper_common.BackendIntEnum(backend), on_gpu=on_gpu
    )
    backend_name = actual_backend.name if hasattr(actual_backend, "name") else actual_backend
    logger.info(f"Using Backend {backend_name} with on_gpu={on_gpu}")
    allocator = model_backends.get_allocator(actual_backend)

    config = turbulence.TurbulenceConfig(
        # -- 1. Numerical parameters (mo_turbdiff_config.f90:124-153)
        impl_s=impl_s,
        impl_t=impl_t,
        imode_tkvmini=imode_tkvmini,
        tkhmin=tkhmin,
        tkmmin=tkmmin,
        tkhmin_strat=tkhmin_strat,
        tkmmin_strat=tkmmin_strat,
        ditsmot=ditsmot,
        imode_frcsmot=imode_frcsmot,
        frcsmot=frcsmot,
        tkesmot=tkesmot,
        frcsecu=frcsecu,
        tkesecu=tkesecu,
        stbsecu=stbsecu,
        prfsecu=prfsecu,
        epsi=epsi,
        it_end=it_end,
        # -- 2. Physical properties of the lower boundary (:155-190)
        rlam_heat=rlam_heat,
        rlam_mom=rlam_mom,
        rat_lam=rat_lam,
        rat_sea=rat_sea,
        rat_glac=rat_glac,
        rat_can=rat_can,
        imode_nsf_wind=imode_nsf_wind,
        rsur_sher=rsur_sher,
        imode_charpar=options.CharnockParameterType(imode_charpar),
        alpha0=alpha0,
        alpha0_max=alpha0_max,
        alpha0_pert=alpha0_pert,
        alpha1=alpha1,
        # -- 3. Stand-ins for external parameter fields (:192-207)
        c_lnd=c_lnd,
        c_sea=c_sea,
        c_soil=c_soil,
        c_stm=c_stm,
        e_surf=e_surf,
        lconst_z0=lconst_z0,
        const_z0=const_z0,
        # -- 4. Stand-ins for dynamical fields (:209-213)
        z0m_dia=z0m_dia,
        z0_ice=z0_ice,
        # -- 5. Turbulent diffusion parameters (:215-254)
        tur_len=tur_len,
        pat_len=pat_len,
        len_min=len_min,
        imode_vel_min=imode_vel_min,
        vel_min=vel_min,
        vel_max=vel_max,
        akt=akt,
        a_heat=a_heat,
        a_mom=a_mom,
        d_heat=d_heat,
        d_mom=d_mom,
        c_diff=c_diff,
        a_stab=a_stab,
        a_hshr=a_hshr,
        clc_diag=clc_diag,
        q_crit=q_crit,
        c_scld=c_scld,
        # -- 6. Switches (:260-284)
        ltkesso=ltkesso,
        ltkecon=ltkecon,
        ltkeshs=ltkeshs,
        ltkenst=ltkenst,
        loutsso=loutsso,
        loutshs=loutshs,
        loutnst=loutnst,
        loutbms=loutbms,
        ltmpcor=ltmpcor,
        lcpfluc=lcpfluc,
        lexpcor=lexpcor,
        lsflcnd=lsflcnd,
        lcirflx=lcirflx,
        ldiff_qi=ldiff_qi,
        ldiff_qs=ldiff_qs,
        lfreeslip=lfreeslip,
        l3dturb=l3dturb,
        # -- 7. Selectors (:290-389)
        imode_tran=imode_tran,
        imode_turb=imode_turb,
        icldm_tran=icldm_tran,
        icldm_turb=options.CloudRepresentationType(icldm_turb),
        itype_wcld=itype_wcld,
        itype_sher=options.ShearProductionType(itype_sher),
        imode_stbcalc=imode_stbcalc,
        ilow_def_cond=ilow_def_cond,
        imode_pat_len=imode_pat_len,
        imode_shshear=imode_shshear,
        imode_tkesso=options.SsoTkeProductionType(imode_tkesso),
        imode_snowsmot=imode_snowsmot,
        itype_2m_diag=itype_2m_diag,
        imode_stadlim=imode_stadlim,
        imode_trancnf=imode_trancnf,
        imode_lamdiff=imode_lamdiff,
        imode_tkemini=imode_tkemini,
        imode_suradap=imode_suradap,
        imode_tkediff=imode_tkediff,
        imode_adshear=imode_adshear,
    )

    metric_state = states.TurbulenceMetricState(
        hhl=hhl,
        dp0=dp0,
        l_hori=l_hori,
        trop_mask=trop_mask,
        innertrop_mask=innertrop_mask,
    )

    grid = grid_wrapper.grid_state.grid
    global granule  # noqa: PLW0603 [global-statement]
    granule = TurbulenceGranule(
        turbulence=turbulence.Turbulence(
            grid=grid,
            config=config,
            params=turbulence.TurbulenceParams(config),
            vertical_grid=grid_wrapper.grid_state.vertical_grid,
            metric_state=metric_state,
            backend=actual_backend,
        ),
        updated_tke=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        num_cells=grid.num_cells,
        unused=_allocate_the_unused_state_fields(grid, allocator),
    )
    if wrapper_config.WAIT_FOR_COMPILATION:
        gtx.wait_for_compilation()


@icon4py_export.export
def turbulence_run(  # noqa: PLR0917 [too-many-positional-arguments]
    # -- the atmospheric column, read-only
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    t: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    prs: fa.CellKField[wpfloat],
    rhoh: fa.CellKField[wpfloat],
    epr: fa.CellKField[wpfloat],
    # -- external forcings of the TKE budget, read-only. The last four are 'REAL(KIND=vp)' in
    #    the Fortran because they come from the mixed-precision dycore diffusion; 'vpfloat' is
    #    'wpfloat' in the double build this port requires, and mypy cannot see through the
    #    runtime rebinding that makes it float32 in a mixed build, hence the four ignores.
    ut_sso: fa.CellKField[wpfloat],
    vt_sso: fa.CellKField[wpfloat],
    hdef2: fa.CellKField[vpfloat],  # type: ignore[valid-type]
    hdiv: fa.CellKField[vpfloat],  # type: ignore[valid-type]
    dwdx: fa.CellKField[vpfloat],  # type: ignore[valid-type]
    dwdy: fa.CellKField[vpfloat],  # type: ignore[valid-type]
    # -- the grid-mean surface state, read-only
    t_g: fa.CellField[wpfloat],
    qv_s: fa.CellField[wpfloat],
    ps: fa.CellField[wpfloat],
    l_pat: fa.CellField[wpfloat],
    # -- turbulence diagnostics. 'gz0', 'tvm', 'tvh', 'tfm', 'tfh', 'tkred_sfc', 'tkred_sfc_h',
    #    'shfl_s' and 'qvfl_s' are read; 'tke', 'tkvm', 'tkvh', 'rcld' and 'rhon' are written.
    gz0: fa.CellField[wpfloat],
    tvm: fa.CellField[wpfloat],
    tvh: fa.CellField[wpfloat],
    tfm: fa.CellField[wpfloat],
    tfh: fa.CellField[wpfloat],
    tkred_sfc: fa.CellField[wpfloat],
    tkred_sfc_h: fa.CellField[wpfloat],
    shfl_s: fa.CellField[wpfloat],
    qvfl_s: fa.CellField[wpfloat],
    tke: fa.CellKField[wpfloat],
    tkvm: fa.CellKField[wpfloat],
    tkvh: fa.CellKField[wpfloat],
    rcld: fa.CellKField[wpfloat],
    rhon: fa.CellKField[wpfloat],
    # -- tendencies. 'tketens' arrives holding the advection tendency and leaves holding the
    #    diffusion tendency; the other five are accumulated onto.
    tketens: fa.CellKField[wpfloat],
    tket_hshr: fa.CellKField[wpfloat],
    u_tens: fa.CellKField[wpfloat],
    v_tens: fa.CellKField[wpfloat],
    t_tens: fa.CellKField[wpfloat],
    qv_tens: fa.CellKField[wpfloat],
    qc_tens: fa.CellKField[wpfloat],
    # -- time steps. ICON passes 'tcall_turb_jg' for both, but the scheme keeps them apart.
    dt_var: gtx.float64,
    dt_tke: gtx.float64,
) -> None:
    """Run one time step of the atmospheric turbulence: 'turbdiff', then 'vertdiff'.

    The argument names are the dummy-argument names of the two ICON calls
    ('mo_nwp_turbdiff_interface.f90:576' and ':672'), so the Fortran side is a transcription.
    'zvari' is not among them: it is granule-internal scratch that carries the conserved
    variables from the first stage to the second, and ICON reads it back only under
    'l_3d_turb_fluxes', which this port does not support.

    'ptr(:)' and 'ndtr' are not among them either, AND THAT ONE IS A SILENT GAP. The ported call
    site runs with 'ndtr = 0' ('test_vertdiff_runs_in_the_configuration_this_port_assumes'
    asserts it against the capture), so the passive-tracer tuples below are empty -- but they
    are empty unconditionally, not because anything here checked. Under 'ldiff_qi', 'ldiff_qs',
    two-moment or SBM microphysics, ART or ComIn tracers, the tracers ICON expects to be
    diffused would simply not be, with no error and no warning. The only thing stopping that is
    the 'nturb_tracer_tot > 0' guard in ICON's 'mo_icon4py_turbulence.f90', which is in another
    repository and invisible from here. See the module docstring.
    """
    if granule is None:
        raise RuntimeError("Turbulence granule not initialized. Call 'turbulence_init' first.")

    input_state = states.TurbulenceInputState(
        u=u,
        v=v,
        t=t,
        qv=qv,
        qc=qc,
        prs=prs,
        rhoh=rhoh,
        epr=epr,
        tke=tke,
        ut_sso=ut_sso,
        vt_sso=vt_sso,
        hdef2=hdef2,
        hdiv=hdiv,
        dwdx=dwdx,
        dwdy=dwdy,
        w=granule.unused["w"],
        tket_conv=granule.unused["tket_conv"],
        # No tracer crosses this boundary. Safe only at 'ndtr = 0', and nothing here can tell:
        # the ICON-side 'nturb_tracer_tot > 0' guard is what makes it safe. Module docstring.
        tracers=(),
    )
    surface_state = states.TurbulenceSurfaceState(
        t_g=t_g,
        qv_s=qv_s,
        ps=ps,
        l_pat=l_pat,
        fr_land=granule.unused["fr_land"],
        l_lake=granule.unused["l_lake"],
        l_sice=granule.unused["l_sice"],
        urb_isa=granule.unused["urb_isa"],
        rlamh_fac=granule.unused["rlamh_fac"],
        z0_waves=granule.unused["z0_waves"],
    )
    # ICON's single 'z_tvs' becomes the granule's two TKE fields; see the module docstring.
    #
    # THIS ONE STAYS A COPY, unlike the surface height in
    # 'Turbulence._derive_what_depends_only_on_the_grid'. Wrapping ICON's 'tke' as
    # 'granule.updated_tke' would cost nothing and would alias exactly the array
    # 'TurbulenceInputState.tke' already points at -- and the two must be distinct fields.
    # 'compute_turbulent_velocity_scale' reads 'previous_velocity_scale=input_state.tke' while
    # writing 'turbulent_velocity_scale=diagnostic_state.updated_tke' in one program, over the
    # whole column, so an alias would feed it values it has just written; and ADR-0001 forbids a
    # physics component writing into its input state, which is why there are two fields at all.
    #
    # Both copies are clamped to 'granule.num_cells' and not to 'tke's own length, which is
    # 'nproma'. 'granule.updated_tke' is 'num_cells' wide, so unclamped this raised a broadcast
    # error the first time the granule ran inside ICON. Clamping the write back also leaves the
    # rows past 'num_cells' -- ICON's block padding, undefined and outside the scheme's column
    # window -- holding exactly what ICON put there.
    granule.updated_tke.ndarray[...] = tke.ndarray[: granule.num_cells, :]  # type: ignore[index]
    diagnostic_state = states.TurbulenceDiagnosticState(
        gz0=gz0,
        tvm=tvm,
        tvh=tvh,
        tfm=tfm,
        tfh=tfh,
        tkred_sfc=tkred_sfc,
        tkred_sfc_h=tkred_sfc_h,
        shfl_s=shfl_s,
        qvfl_s=qvfl_s,
        tkvm=tkvm,
        tkvh=tkvh,
        rcld=rcld,
        rhon=rhon,
        updated_tke=granule.updated_tke,
        tfv=granule.unused["tfv"],
        tprn=granule.unused["tprn"],
        edr=granule.unused["edr"],
        tur_len_scale=granule.unused["tur_len_scale"],
        tcm=granule.unused["tcm"],
        tch=granule.unused["tch"],
        umfl_s=granule.unused["umfl_s"],
        vmfl_s=granule.unused["vmfl_s"],
        t_2m=granule.unused["t_2m"],
        qv_2m=granule.unused["qv_2m"],
        td_2m=granule.unused["td_2m"],
        rh_2m=granule.unused["rh_2m"],
        u_10m=granule.unused["u_10m"],
        v_10m=granule.unused["v_10m"],
    )
    tendency_state = states.TurbulenceTendencyState(
        ddt_tke=tketens,
        tket_hshr=tket_hshr,
        ddt_u=u_tens,
        ddt_v=v_tens,
        ddt_t=t_tens,
        ddt_qv=qv_tens,
        ddt_qc=qc_tens,
        # As 'input_state.tracers': empty unconditionally, guarded only from the ICON side.
        ddt_tracers=(),
    )

    granule.turbulence.run(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        dt_var=dt_var,
        dt_tke=dt_tke,
    )

    tke.ndarray[: granule.num_cells, :] = granule.updated_tke.ndarray  # type: ignore[index]

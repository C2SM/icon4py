# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end datatest of 'Turbulence.run': 'turbdiff-entry' through to 'vertdiff-exit'.

`Turbulence.run` is the unit 'mo_nwp_turbdiff_interface.f90' substitutes. That interface calls
'turbdiff' at :576 and 'vertdiff' at :672 with nothing between them but a timer stop and a timer
start, and the two calls share six arrays. This module runs both stages from the state ICON
handed the first one and compares what comes out against the state ICON handed back after the
second, for all four timesteps of exp.mch_icon-ch2_small.

'test_turbdiff_granule.py' does the same for the first stage alone and 'test_vertdiff.py' for
the second, program by program. What neither of them can see is the handover.

WHAT ONLY THIS MODULE CAN CATCH
-------------------------------
* 'zvari' IS A CROSS-STAGE ALIAS AND THE SECOND STAGE DESTROYS IT. The interface passes one
  'zvari(:,:,0:5)' to both calls; 'turbdiff' leaves the vertical gradients of the
  quasi-conserved variables in components 1..5 and 'vertdiff' overwrites every one of them with
  the right-hand side of the variable that shares the index. Measured across the two exit
  savepoints: components 1 to 4 differ in all 670356 values and component 5 in 68684 of them.
  A port that gave 'vertdiff' fresh buffers would pass both stage tests and still not be the
  scheme.
* 'rhon' IS WRITTEN BY BOTH, IN DIFFERENT ROWS. 'turbdiff' interpolates it onto the half levels
  and puts the Prandtl-layer boundary value in the surface row; 'vertdiff' replaces that row
  with the ideal-gas density of the ground (turb_vertdiff.f90:536-542), which is a different
  quantity and the Fortran says so. It is also what turns 'rhon' from the near-miss of
  'test_turbdiff_granule' -- 5.5e-16 on 14 to 19 values, all in that one row -- into a
  bit-exact output of the pair.
* THE SECOND STAGE CONSUMES THE FIRST'S DIFFUSION COEFFICIENTS, including the surface row
  section 4) does not write, and it must not read the model top. See
  'test_the_model_top_of_the_diffusion_coefficients_is_never_read'.
* ADR-0001 over the pair: the granule must never write 'u', 'v', 't', 'qv' or 'qc'. Neither does
  the Fortran -- '597f090cf2' removed the optional in-place incrementation, so 'turbdiff' and
  'vertdiff' write tendencies only -- and 'test_run_never_writes_the_state_it_was_given'
  measures it.
* THE CALLER'S FIELDS ARE WIDER THAN THE GRID, which is the one thing every other test in this
  package gets wrong by building both kinds at 'grid.num_cells'. See the last section of this
  module.

WHERE THE ENTRY STATE COMES FROM, AND WHY THAT IS SOUND
-------------------------------------------------------
'turbdiff-entry' does not carry every field the pair needs: the two surface flux densities that
are 'vertdiff's lower boundary condition and the 'qv' and 'qc' tendencies are not arguments of
'turbdiff' at all, and are serialized at 'vertdiff-entry' instead. Taking them from there is
only legitimate if 'turbdiff' cannot have changed them in between, and
'test_the_two_stages_meet_on_the_state_icon_hands_between_them' measures exactly that: every
field the interface carries from one call to the other is BIT-IDENTICAL at 'turbdiff-exit' and
at 'vertdiff-entry', on all four dates. That test runs no stencil and is the precondition for
everything below.

WHY THE 'run' TESTS XFAIL ON 'embedded'
----------------------------------------
The first stage contains five programs that select a boundary row with 'concat_where', which
gt4py 1.1.10 cannot execute there, so the composed run is marked 'uses_concat_where' and
xfails before it starts -- as 'test_turbdiff_granule' does. The 'run_vertdiff' tests carry
'embedded_too_slow' instead: 'vertdiff' uses no 'concat_where' at all, but its solve is two
'scan_operator's that the embedded backend runs as a Python loop over every column and level.

THE FOUR GATES
--------------
Two of them are new to this module, one is 'test_turbdiff_granule's and one belongs to a single
stencil. They are apart rather than merged because the errors differ by four decades and for one
reason: 'turbdiff' has exactly two inexact steps (section 0)'s 'EXP' and section 2a)'s
'EXP(2/3*LOG())') and 'vertdiff' has one ('EXP(rdocp*LOG(ps/p0))'), and what separates the
groups is not the arithmetic but whether the quantity is a DIFFERENCE OF TWO NEARLY EQUAL
NUMBERS. Every tendency is: 'dvar_at' as the scheme found it plus an increment several decades
smaller. So a relative error of 1e-7 in 't_tens' sits on an ABSOLUTE error of 8.4e-15, which is
a rounding of a quantity of order 1e-4, and giving the other twenty outputs that tolerance would
mean widening them by six decades for a reason that does not apply to them.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence,
    turbulence_states as states,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403
from .test_turbdiff_granule import CONFIG, PARAMS


#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), zero-based as the reader takes them.
PRESSURE, U_M, V_M, TET_L, H2O_G, LIQ = 0, 1, 2, 3, 4, 5

#: Gate key for every output of the pair that is not a tendency.
GRANULE_GATE = "run"

#: Gate key for the wind, moisture and cloud-water tendencies, and for the diffusion increment
#: they are formed from. See "The four gates" above for why these are not `GRANULE_GATE`.
TENDENCY_GATE = "run_tendencies"

#: Gate key for the temperature tendency alone, which needs three decades more than the other
#: four. Not because its arithmetic is worse -- 'compute_and_apply_potential_temperature_-
#: diffusion_tendency' is gated 'Exact()' and the stage produces 't_tens' bit-exactly when it is
#: run from ICON's own state. It is the accumulation: 't_tens' arrives holding the tendency of
#: whatever ran before and the diffusion increment is added to it, and where the two nearly
#: cancel the relative error of the sum is the absolute error divided by the remainder.
TEMPERATURE_TENDENCY_GATE = "run_temperature_tendency"

#: The TKE tendency keeps the gate 'test_turbdiff_granule' measured for it; 'vertdiff' does not
#: touch it and the number is unchanged by the composition (4.0346e-09 at 06:01:00, both there
#: and here).
TKE_TENDENCY_GATE = "run_turbdiff_tke_tendency"


#: What comes out of BOTH stages with ICON's bits, measured on 'gtfn_cpu' and 'dace_cpu' at all
#: four dates, each with the reason it escapes the three transcendentals of the pair.
BIT_EXACT_END_TO_END: tuple[tuple[str, str], ...] = (
    (
        "rhon",
        "The interior rows are the mass-weighted interpolation of ICON's own 'rhoh' and the "
        "surface row is 'ps/(R_d*(1 + rvd_m_o*qv_s)*t_g)', which 'vertdiff' writes over what "
        "'turbdiff' left there. Neither passes through an exponential -- and this is the "
        "output that is NOT bit-exact after the first stage alone.",
    ),
    (
        "disc_mom",
        "'rho*dz/dt' is a difference of 'hhl' times ICON's 'rhoh' times a reciprocal formed "
        "once, and nothing upstream of it is computed by the granule at all.",
    ),
    (
        "diff_dep",
        "The interior rows are half-sums of layer depths, and the surface row is "
        "'tkvh(:,ke1)/tvh' -- the one row of the diffusion coefficient that 'turbdiff' does "
        "NOT write, so it is still 'turbtran's.",
    ),
    (
        "cur_prof [qc]",
        "Cloud water is diffused in the units it arrives in and its lower boundary value is "
        "the literal zero of 'ilow_def_cond = 2', so the profile is ICON's own 'qc'.",
    ),
)


class Turbulence(NamedTuple):
    """One timestep run through both stages: the granule, the states it wrote and the oracle."""

    granule: turbulence.Turbulence
    entry: sb.IconTurbdiffEntrySavepoint
    after_turbdiff: sb.IconTurbulenceSavepoint
    after_vertdiff: sb.IconVertdiffExitSavepoint
    diagnostic_state: states.TurbulenceDiagnosticState
    tendency_state: states.TurbulenceTendencyState
    input_state: states.TurbulenceInputState
    surface_state: states.TurbulenceSurfaceState
    nlev: int
    #: Half-open range of columns the scheme computed; every comparison below is masked with it.
    columns: slice


def _unused_cell_field(grid, backend) -> gtx.Field:
    """A cell field for a state member neither stage reads."""
    return data_alloc.zero_field(grid, dims.CellDim, allocator=backend)


def _unused_cell_k_field(grid, backend) -> gtx.Field:
    """A (Cell, K) field on half levels for a state member neither stage reads."""
    return data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )


def _states_for_both_stages(data_provider, icon_grid, grid_savepoint, date, backend):
    """Build the granule and its five state containers from the state ICON gave 'turbdiff'.

    'test_turbdiff_granule' builds a deliberately narrower version of this: it zero-fills every
    member 'turbdiff' does not read, so that a stray read there shows up as a wrong answer. Here
    those members are real, because the second stage does read them -- and the four that
    'turbdiff-entry' does not carry ('shfl_s', 'qvfl_s' and the 'qv' and 'qc' tendencies) come
    from 'vertdiff-entry', which is sound only because
    'test_the_two_stages_meet_on_the_state_icon_hands_between_them' shows the two savepoints
    agree on everything they have in common.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    section_0 = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    before_vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)
    nlev = entry.ke()

    metric_state = states.TurbulenceMetricState(
        hhl=utils.copy_of(entry.hhl(), backend),
        dp0=utils.copy_of(entry.dp0(), backend),
        l_hori=utils.copy_of(entry.l_hori(), backend),
        trop_mask=utils.copy_of(entry.trop_mask(), backend),
        innertrop_mask=utils.copy_of(entry.innertrop_mask(), backend),
    )
    input_state = states.TurbulenceInputState(
        u=utils.copy_of(entry.u(), backend),
        v=utils.copy_of(entry.v(), backend),
        w=utils.copy_of(entry.w(), backend),
        t=utils.copy_of(entry.t(), backend),
        qv=utils.copy_of(entry.qv(), backend),
        qc=utils.copy_of(entry.qc(), backend),
        prs=utils.copy_of(entry.prs(), backend),
        rhoh=utils.copy_of(entry.rhoh(), backend),
        epr=utils.copy_of(entry.epr(), backend),
        tke=utils.copy_of(entry.tke(), backend),
        # 'ndtr = 0' at this call site, so 'vertdiff' has no passive tracers to diffuse.
        tracers=(),
        ut_sso=utils.copy_of(entry.ut_sso(), backend),
        vt_sso=utils.copy_of(entry.vt_sso(), backend),
        tket_conv=utils.copy_of(entry.tket_conv(), backend),
        hdef2=utils.copy_of(entry.hdef2(), backend),
        hdiv=utils.copy_of(entry.hdiv(), backend),
        dwdx=utils.copy_of(entry.dwdx(), backend),
        dwdy=utils.copy_of(entry.dwdy(), backend),
    )
    surface_state = states.TurbulenceSurfaceState(
        t_g=utils.copy_of(entry.t_g(), backend),
        qv_s=utils.copy_of(entry.qv_s(), backend),
        ps=utils.copy_of(entry.ps(), backend),
        l_pat=utils.copy_of(entry.l_pat(), backend),
        # Read by 'turbtran' only; carried so that the container is complete.
        fr_land=_unused_cell_field(icon_grid, backend),
        l_lake=data_alloc.zero_field(icon_grid, dims.CellDim, dtype=bool, allocator=backend),
        l_sice=data_alloc.zero_field(icon_grid, dims.CellDim, dtype=bool, allocator=backend),
        urb_isa=_unused_cell_field(icon_grid, backend),
        rlamh_fac=_unused_cell_field(icon_grid, backend),
        z0_waves=_unused_cell_field(icon_grid, backend),
    )
    diagnostic_state = states.TurbulenceDiagnosticState(
        gz0=utils.copy_of(entry.gz0(), backend),
        tvm=utils.copy_of(entry.tvm(), backend),
        tvh=utils.copy_of(entry.tvh(), backend),
        tfm=utils.copy_of(entry.tfm(), backend),
        tfh=utils.copy_of(entry.tfh(), backend),
        tfv=utils.copy_of(entry.tfv(), backend),
        tkred_sfc=utils.copy_of(entry.tkred_sfc(), backend),
        tkred_sfc_h=utils.copy_of(entry.tkred_sfc_h(), backend),
        tkvm=utils.copy_of(entry.tkvm(), backend),
        tkvh=utils.copy_of(entry.tkvh(), backend),
        rcld=utils.copy_of(entry.rcld(), backend),
        # 'rhon' and the updated TKE are pure outputs of the pair with no entry state, so they
        # are NaN-filled: a row neither stage writes must not hold a plausible value.
        rhon=utils.nan_like(entry.rcld(), backend),
        updated_tke=utils.nan_like(entry.tke(), backend),
        # 'vertdiff's lower boundary condition for temperature and water vapour. Written by
        # 'turbtran' and aggregated by the surface scheme, so it is an input to both stages
        # here; not an argument of 'turbdiff' and therefore not at its savepoint.
        shfl_s=utils.copy_of(before_vertdiff.shfl_s(), backend),
        qvfl_s=utils.copy_of(before_vertdiff.qvfl_s(), backend),
        # Written by 'turbtran', read by neither stage here.
        tcm=_unused_cell_field(icon_grid, backend),
        tch=_unused_cell_field(icon_grid, backend),
        umfl_s=_unused_cell_field(icon_grid, backend),
        vmfl_s=_unused_cell_field(icon_grid, backend),
        t_2m=_unused_cell_field(icon_grid, backend),
        qv_2m=_unused_cell_field(icon_grid, backend),
        td_2m=_unused_cell_field(icon_grid, backend),
        rh_2m=_unused_cell_field(icon_grid, backend),
        u_10m=_unused_cell_field(icon_grid, backend),
        v_10m=_unused_cell_field(icon_grid, backend),
        # Not written in this configuration: 'tprn' is a '(1,1)' dummy, and 'edr' and
        # 'tur_len_scale' are disassociated unless 'ldiagnose_tke'.
        tprn=_unused_cell_k_field(icon_grid, backend),
        edr=_unused_cell_k_field(icon_grid, backend),
        tur_len_scale=_unused_cell_k_field(icon_grid, backend),
    )
    tendency_state = states.TurbulenceTendencyState(
        # All six are 'INTENT(INOUT)': what is already there is the tendency of whatever ran
        # before, and the scheme adds to it. Starting them at zero would make the accumulation
        # untestable, which is the point of taking them from the savepoints.
        ddt_tke=utils.copy_of(entry.tketens(), backend),
        ddt_u=utils.copy_of(before_vertdiff.u_tens(), backend),
        ddt_v=utils.copy_of(before_vertdiff.v_tens(), backend),
        ddt_t=utils.copy_of(before_vertdiff.t_tens(), backend),
        ddt_qv=utils.copy_of(before_vertdiff.qv_tens(), backend),
        ddt_qc=utils.copy_of(before_vertdiff.qc_tens(), backend),
        ddt_tracers=(),
        # 'tket_hshr' is not serialized at 'turbdiff-entry' -- it is an output slot the entry
        # hook does not carry -- so its entry state is read from 'turbdiff-0-exit', which
        # sections 0) and 1) leave untouched.
        tket_hshr=utils.copy_of_raw_field(section_0, "td_tket_hshr", backend),
    )
    vertical_grid = v_grid.VerticalGrid(
        config=v_grid.VerticalGridConfig(num_levels=nlev),
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )
    granule = turbulence.Turbulence(
        grid=icon_grid,
        config=CONFIG,
        params=PARAMS,
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        backend=backend,
    )
    return granule, input_state, surface_state, diagnostic_state, tendency_state, entry


def _run(data_provider, icon_grid, grid_savepoint, date: str, backend) -> Turbulence:
    """Run both stages once, in the order the ICON interface calls them."""
    granule, input_state, surface_state, diagnostic, tendency, entry = _states_for_both_stages(
        data_provider, icon_grid, grid_savepoint, date, backend
    )
    before_vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)
    granule.run(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        dt_var=before_vertdiff.dt_var(),
        dt_tke=entry.dt_tke(),
    )
    return Turbulence(
        granule=granule,
        entry=entry,
        after_turbdiff=data_provider.from_savepoint_turbdiff_exit(date=date),
        after_vertdiff=data_provider.from_savepoint_vertdiff_exit(date=date),
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        input_state=input_state,
        surface_state=surface_state,
        nlev=entry.ke(),
        columns=slice(entry.ivstart(), entry.ivend()),
    )


def _run_the_second_stage_alone(
    data_provider, icon_grid, grid_savepoint, date: str, backend, *, poison_the_model_top=False
) -> Turbulence:
    """Run 'run_vertdiff' from ICON's own 'vertdiff-entry' state, without running 'turbdiff'.

    The granule has no way to be handed 'zvari' or 'rhon' from outside -- they are its own
    fields, which is the whole point of them -- so the state is written into them here. That is
    white-box, and it buys the only comparison in the package that puts the SECOND stage against
    ICON with the FIRST stage's rounding removed: run this way every output but the two the
    surface Exner factor reaches is bit-exact, which is what makes the composed tolerances
    attributable to 'turbdiff' and not to 'vertdiff'.

    'poison_the_model_top' writes NaN into 'tkvm(:,0)' and 'tkvh(:,0)'. Those rows are exactly
    zero in the reference data, so anything multiplying by them produces exactly zero and a
    comparison against ICON cannot tell a program that correctly skipped the row from one that
    wrongly wrote it. NaN can: it propagates through zero.
    """
    granule, input_state, surface_state, diagnostic, tendency, entry = _states_for_both_stages(
        data_provider, icon_grid, grid_savepoint, date, backend
    )
    before = data_provider.from_savepoint_vertdiff_entry(date=date)
    for target, source in (
        (diagnostic.rhon, before.rhon()),
        (diagnostic.tkvm, before.tkvm()),
        (diagnostic.tkvh, before.tkvh()),
        (diagnostic.tvm, before.tvm()),
        (diagnostic.tvh, before.tvh()),
        (granule._gradient_zonal_wind, before.zvari(U_M)),
        (granule._gradient_meridional_wind, before.zvari(V_M)),
        (granule._gradient_liquid_water_potential_temperature, before.zvari(TET_L)),
        (granule._gradient_total_water, before.zvari(H2O_G)),
        (granule._gradient_liquid_water, before.zvari(LIQ)),
    ):
        utils.overwrite_with(target, source, backend)
    if poison_the_model_top:
        diagnostic.tkvm.ndarray[:, 0] = np.nan
        diagnostic.tkvh.ndarray[:, 0] = np.nan
    granule.run_vertdiff(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        dt_var=before.dt_var(),
    )
    return Turbulence(
        granule=granule,
        entry=entry,
        after_turbdiff=data_provider.from_savepoint_turbdiff_exit(date=date),
        after_vertdiff=data_provider.from_savepoint_vertdiff_exit(date=date),
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        input_state=input_state,
        surface_state=surface_state,
        nlev=entry.ke(),
        columns=slice(entry.ivstart(), entry.ivend()),
    )


def _second_stage_outputs(run: Turbulence) -> tuple[tuple[str, gtx.Field, gtx.Field, slice], ...]:
    """Everything 'vertdiff' produces, against 'vertdiff-exit', with the rows ICON defines.

    The row windows narrower than the whole column are the seven leftover rows
    'test_vertdiff.py::test_the_unwritten_workspace_rows_are_leftovers_not_results' argues
    'vertdiff' never writes.
    """
    nlev, granule, after = run.nlev, run.granule, run.after_vertdiff
    tendency, diagnostic = run.tendency_state, run.diagnostic_state
    everything = slice(0, nlev + 1)
    main_levels = slice(0, nlev)
    return (
        # -- what the scheme is for
        ("u_tens", tendency.ddt_u, after.u_tens(), main_levels),
        ("v_tens", tendency.ddt_v, after.v_tens(), main_levels),
        ("t_tens", tendency.ddt_t, after.t_tens(), main_levels),
        ("qv_tens", tendency.ddt_qv, after.qv_tens(), main_levels),
        ("qc_tens", tendency.ddt_qc, after.qc_tens(), main_levels),
        # 'rhon(:,1)' is written by NEITHER stage: 'bound_level_interp' starts at 'k=2',
        # 'adjust_satur_equil' supplies 'ke1' and 'vertdiff' rewrites 'ke1' alone, so the model
        # top holds untouched memory (around -0.025 here) on ICON's side of the comparison.
        ("rhon", diagnostic.rhon, after.rhon(), slice(1, nlev + 1)),
        # -- 'zvari', which the second stage overwrites with its right-hand sides
        ("zvari(:,:,u_m)", granule._gradient_zonal_wind, after.zvari(U_M), everything),
        ("zvari(:,:,v_m)", granule._gradient_meridional_wind, after.zvari(V_M), everything),
        (
            "zvari(:,:,tet_l)",
            granule._gradient_liquid_water_potential_temperature,
            after.zvari(TET_L),
            everything,
        ),
        ("zvari(:,:,h2o_g)", granule._gradient_total_water, after.zvari(H2O_G), everything),
        ("zvari(:,:,liq)", granule._gradient_liquid_water, after.zvari(LIQ), everything),
        # -- the workspace the exit hook carries for localisation
        ("disc_mom", granule._discretisation_momentum, after.disc_mom(), main_levels),
        ("expl_mom", granule._diffusion_momentum, after.expl_mom(), slice(1, nlev + 1)),
        ("impl_mom", granule._implicit_diffusion_momentum, after.impl_mom(), slice(1, nlev + 1)),
        ("invs_mom", granule._inverted_diffusion_momentum, after.invs_mom(), main_levels),
        ("invs_fac", granule._inversion_factor, after.invs_fac(), slice(1, nlev)),
        ("diff_dep", granule._diffusion_depth, after.diff_dep(), slice(1, nlev + 1)),
        ("cur_prof [qc]", granule._current_profile, after.cur_prof(), everything),
        ("dif_tend [qc]", granule._diffusion_increment, after.dif_tend(), main_levels),
    )


def _first_stage_outputs(run: Turbulence) -> tuple[tuple[str, gtx.Field, gtx.Field, slice], ...]:
    """What 'turbdiff' produced and 'vertdiff' must leave alone, against 'turbdiff-exit'.

    'tkvm' and 'tkvh' are the interesting pair: 'vertdiff' declares them 'INTENT(INOUT)' and
    reads them at every flux level including the surface row, and writes neither.
    """
    nlev, after = run.nlev, run.after_turbdiff
    diagnostic, tendency = run.diagnostic_state, run.tendency_state
    everything = slice(0, nlev + 1)
    return (
        ("tke", diagnostic.updated_tke, after.tke(), everything),
        ("tkvm", diagnostic.tkvm, after.tkvm(), everything),
        ("tkvh", diagnostic.tkvh, after.tkvh(), everything),
        ("rcld", diagnostic.rcld, after.rcld(), everything),
        ("tketens", tendency.ddt_tke, after.tketens(), everything),
        ("tket_hshr", tendency.tket_hshr, after.tket_hshr(), everything),
    )


def _gate_for(quantity: str) -> str:
    """Which of the four gates an output of the pair is compared under."""
    if quantity == "tketens":
        return TKE_TENDENCY_GATE
    if quantity == "t_tens":
        return TEMPERATURE_TENDENCY_GATE
    if quantity.endswith("_tens") or quantity == "dif_tend [qc]":
        return TENDENCY_GATE
    return GRANULE_GATE


# ------------------------------------------------- the handover, before any stencil runs ---


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_two_stages_meet_on_the_state_icon_hands_between_them(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Every field the ICON interface carries from 'turbdiff' to 'vertdiff' is unchanged.

    This is the precondition for `Turbulence.run` being two calls and nothing else, and for this
    module taking four of its entry fields from the second savepoint rather than the first. It
    is measured rather than read off the interface, because a statement between the two calls --
    a unit conversion, an aggregation, a tendency reset -- would be invisible in the source of
    either subroutine and fatal to the composition.

    Nineteen fields plus all six 'zvari' components, bit for bit over the computed columns. Six
    of them are outputs of the first stage ('rhon', 'tkvm', 'tkvh' and the gradients) and the
    rest are inputs both stages read; the three tendencies 'turbdiff' also declares
    'INTENT(INOUT)' are here too, since a write there is exactly what would make the accumulated
    'u_tens' of this module wrong.
    """
    before = data_provider.from_savepoint_turbdiff_entry(date=date)
    after_turbdiff = data_provider.from_savepoint_turbdiff_exit(date=date)
    before_vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)
    columns = slice(before.ivstart(), before.ivend())

    shared: list[tuple[str, object, object]] = [
        # written by 'turbdiff', read by 'vertdiff'
        ("rhon", after_turbdiff.rhon(), before_vertdiff.rhon()),
        ("tkvm", after_turbdiff.tkvm(), before_vertdiff.tkvm()),
        ("tkvh", after_turbdiff.tkvh(), before_vertdiff.tkvh()),
        # read by both, written by neither
        ("u", before.u(), before_vertdiff.u()),
        ("v", before.v(), before_vertdiff.v()),
        ("t", before.t(), before_vertdiff.t()),
        ("qv", before.qv(), before_vertdiff.qv()),
        ("qc", before.qc(), before_vertdiff.qc()),
        ("prs", before.prs(), before_vertdiff.prs()),
        ("rhoh", before.rhoh(), before_vertdiff.rhoh()),
        ("epr", before.epr(), before_vertdiff.epr()),
        ("hhl", before.hhl(), before_vertdiff.hhl()),
        ("ps", before.ps(), before_vertdiff.ps()),
        ("t_g", before.t_g(), before_vertdiff.t_g()),
        ("qv_s", before.qv_s(), before_vertdiff.qv_s()),
        ("tvm", before.tvm(), before_vertdiff.tvm()),
        ("tvh", before.tvh(), before_vertdiff.tvh()),
        # 'INTENT(INOUT)' in the first stage and accumulated onto by the second
        ("u_tens", before.u_tens(), before_vertdiff.u_tens()),
        ("v_tens", before.v_tens(), before_vertdiff.v_tens()),
        ("t_tens", before.t_tens(), before_vertdiff.t_tens()),
    ]
    shared += [
        (
            f"zvari(:,:,{component})",
            after_turbdiff.zvari(component),
            before_vertdiff.zvari(component),
        )
        for component in range(6)
    ]
    for quantity, leaving_turbdiff, entering_vertdiff in shared:
        got = data_alloc.as_numpy(leaving_turbdiff)[columns]
        want = data_alloc.as_numpy(entering_vertdiff)[columns]
        assert np.array_equal(got, want), (
            f"'{quantity}' differs between 'turbdiff-exit' and 'vertdiff-entry' in "
            f"{np.count_nonzero(got != want)} of {got.size} values, so something between the "
            "two ICON calls touches it and 'Turbulence.run' is not the two calls alone."
        )

    # The two stages really are given the same time step, which is why 'run' passing 'dt_var'
    # and 'dt_tke' separately is a faithfulness choice and not a difference.
    assert before.dt_tke() == before_vertdiff.dt_var()


# ------------------------------------------------------ the second stage, on its own ---


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_run_vertdiff_is_bit_exact_when_it_starts_from_icons_own_state(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """'run_vertdiff' alone reproduces ICON exactly, save for its one transcendental.

    Eighteen of the twenty quantities the exit savepoint holds come out BIT-EXACT on all four
    dates, so this test is ungated: the stage needs no tolerance of its own and every tolerance
    the composed run needs is attributable to the first stage.

    The two exceptions are the surface Exner factor 'eprs = EXP(rdocp*LOG(ps/p0ref))' and the
    single value of 'zvari(:,ke1,tet_l)' it reaches, both of which
    'test_vertdiff.py::test_the_surface_exner_factor_is_the_only_transcendental' measures
    against ICON; they are checked below with the counts that test established rather than with
    a tolerance, so that a third affected value would fail here.

    It is also what puts `_setup_vertdiff_programs`' twenty-two vertical domains and
    `_allocate_the_vertdiff_working_set`'s eleven fields against ICON. 'test_vertdiff.py' runs
    the same eighteen stencils but chooses their domains and buffers itself.
    """
    run = _run_the_second_stage_alone(data_provider, icon_grid, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns

    for quantity, computed, reference, levels in _second_stage_outputs(run):
        got = data_alloc.as_numpy(computed)[columns, levels]
        want = data_alloc.as_numpy(reference)[columns, levels]
        if quantity == "zvari(:,:,tet_l)":
            continue  # the surface Exner factor reaches one value of it; see below
        assert np.array_equal(got, want), (
            f"'{quantity}' is bit-exact against ICON when 'run_vertdiff' starts from ICON's own "
            f"state and is not any more: max abs {np.nanmax(np.abs(got - want))} over "
            f"{np.count_nonzero(got != want)} of {got.size} values."
        )

    # The one transcendental, and exactly how far it travels. 'eprs' is declared
    # '(nvec, ke1:ke1)' in the Fortran and the savepoint hands it back as a cell field, so the
    # surface row of the granule's half-level field is what it has to be compared against.
    exner = data_alloc.as_numpy(run.granule._surface_exner_factor)[:, nlev]
    utils.assert_agrees_with_icon(
        "compute_surface_air_density_and_exner_factor",
        "eprs",
        exner,
        run.after_vertdiff.eprs(),
        columns=columns,
    )
    temperature = data_alloc.as_numpy(run.granule._gradient_liquid_water_potential_temperature)[
        columns
    ]
    icon = data_alloc.as_numpy(run.after_vertdiff.zvari(TET_L))[columns]
    assert np.array_equal(temperature[:, :nlev], icon[:, :nlev]), (
        "'zvari(:,:,tet_l)' is bit-exact on the diffused rows: the Exner factor can only reach "
        "the surface row, through the temperature's lower boundary value."
    )
    deviating = np.count_nonzero(temperature[:, nlev] != icon[:, nlev])
    assert deviating <= 1, (
        f"the surface Exner factor moves 'zvari(:,ke1,tet_l)' in {deviating} of "
        f"{temperature.shape[0]} columns; measured at most one on all four dates."
    )


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_model_top_of_the_diffusion_coefficients_is_never_read(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """'vertdiff' must not read 'tkvm(:,0)' or 'tkvh(:,0)', and a comparison cannot show it.

    The model-top row of both diffusion coefficients is exactly zero in the reference data, so
    every product with it is exactly zero and a comparison against ICON passes whether or not a
    program reached that row. Section 6) of 'turbdiff' had exactly this hole and it was closed
    with INPUT poison rather than output poison: NaN into the row, which no arithmetic can turn
    back into a number.

    'vertdiff' consumes the same coefficients -- 'vtyp(mom)%tkv => tkvm' and
    'vtyp(sca)%tkv => tkvh' (turb_vertdiff.f90:503-504) -- and 'vert_grad_diff' reads them from
    'k_hi+1' to 'k_sf', never at 'k_tp+1'. Fifteen outputs, poisoned against clean, on all four
    dates.
    """
    clean = _run_the_second_stage_alone(data_provider, icon_grid, grid_savepoint, date, backend)
    poisoned = _run_the_second_stage_alone(
        data_provider, icon_grid, grid_savepoint, date, backend, poison_the_model_top=True
    )
    columns = clean.columns
    assert np.all(np.isnan(data_alloc.as_numpy(poisoned.diagnostic_state.tkvm)[columns, 0])), (
        "the poison did not reach the field the granule read."
    )
    for (quantity, computed, _, levels), (_, poisoned_field, _, _) in zip(
        _second_stage_outputs(clean), _second_stage_outputs(poisoned), strict=True
    ):
        got = data_alloc.as_numpy(computed)[columns, levels]
        contaminated = data_alloc.as_numpy(poisoned_field)[columns, levels]
        assert np.array_equal(got, contaminated), (
            f"'{quantity}' changed when 'tkvm(:,0)' and 'tkvh(:,0)' were poisoned, so "
            f"'run_vertdiff' reads the model-top row of the diffusion coefficients: "
            f"{np.count_nonzero(got != contaminated)} of {got.size} values differ."
        )


# ------------------------------------------------------------------- both stages ---


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_run_reproduces_the_state_icon_returns(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """Every quantity the pair returns agrees with ICON under the granule's gates.

    Nineteen outputs against 'vertdiff-exit' and six against 'turbdiff-exit'. The second group
    is what 'vertdiff' must leave exactly as it found it, and comparing it here rather than only
    in 'test_turbdiff_granule' is what would catch a second stage that wrote 'tkvm' or reset a
    tendency.

    This needs 'run', 'run_tendencies' and 'run_temperature_tendency' in 'gate_registry.GATES';
    without them 'gate_for' raises 'UnregisteredStencilError' rather than defaulting to
    'Exact()', which is the registry working as intended.
    """
    run = _run(data_provider, icon_grid, grid_savepoint, date, backend)
    for quantity, computed, reference, levels in (
        *_second_stage_outputs(run),
        *_first_stage_outputs(run),
    ):
        utils.assert_agrees_with_icon(
            _gate_for(quantity),
            quantity,
            computed,
            reference,
            columns=run.columns,
            levels=levels,
        )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_quantities_that_survive_both_stages_bit_exactly(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """The outputs that reach ICON's numbers exactly through sixty programs, asserted ungated.

    A single end-to-end tolerance would hide this. 'rhon' is the one worth the test on its own:
    it is NOT bit-exact after the first stage -- 'test_turbdiff_granule' records it as the
    instructive near-miss, 5.5e-16 on 14 to 19 values, all in the surface row -- and the second
    stage overwrites exactly that row with an expression that has no exponential in it. So the
    pair is exact where either stage alone is not, and only a composed test can say so.
    """
    run = _run(data_provider, icon_grid, grid_savepoint, date, backend)
    exact = dict(BIT_EXACT_END_TO_END)
    compared = set()
    for quantity, computed, reference, levels in _second_stage_outputs(run):
        if quantity not in exact:
            continue
        compared.add(quantity)
        got = data_alloc.as_numpy(computed)[run.columns, levels]
        want = data_alloc.as_numpy(reference)[run.columns, levels]
        assert np.array_equal(got, want), (
            f"'{quantity}' was bit-exact through both stages when the gates were measured and "
            f"is not any more: max abs {np.nanmax(np.abs(got - want))} over "
            f"{np.count_nonzero(got != want)} of {got.size} values. {exact[quantity]}"
        )
    assert compared == set(exact), f"not every claim was checked: {set(exact) - compared}"


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_second_stage_overwrites_the_gradients_the_first_left(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """'zvari(:,:,1..5)' comes out holding right-hand sides, not the vertical gradients.

    The one cross-stage alias, and the one place where "both stages passed their own test" is
    not enough. The two stages are run separately here so that the state BETWEEN them can be
    kept -- which is what makes this a measurement of the handover rather than of the answer,
    and lets it be asserted without any tolerance at all.

    Component 0 is the opposite claim. It is the half-level pressure and then section 3)'s
    circulation acceleration, and 'vertdiff' has no variable with index 0, so it must come
    through the second stage untouched.

    ICON's own two exit savepoints are checked first, so that neither claim can be vacuous: if
    'turbdiff-exit' and 'vertdiff-exit' agreed on a component, agreeing with the second would
    say nothing about which stage wrote it.
    """
    before_vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)
    after_turbdiff = data_provider.from_savepoint_turbdiff_exit(date=date)
    after_vertdiff = data_provider.from_savepoint_vertdiff_exit(date=date)
    granule, input_state, surface_state, diagnostic, tendency, entry = _states_for_both_stages(
        data_provider, icon_grid, grid_savepoint, date, backend
    )
    columns = slice(entry.ivstart(), entry.ivend())
    components = (
        (U_M, granule._gradient_zonal_wind),
        (V_M, granule._gradient_meridional_wind),
        (TET_L, granule._gradient_liquid_water_potential_temperature),
        (H2O_G, granule._gradient_total_water),
        (LIQ, granule._gradient_liquid_water),
    )

    # What ICON does with the storage, so that the claims below are about something.
    for component, _ in components:
        gradients = data_alloc.as_numpy(after_turbdiff.zvari(component))[columns]
        right_hand_sides = data_alloc.as_numpy(after_vertdiff.zvari(component))[columns]
        moved = np.count_nonzero(gradients != right_hand_sides)
        assert moved > gradients.size // 100, (
            f"ICON's own 'zvari(:,:,{component})' barely moves between the two exit savepoints "
            f"({moved} of {gradients.size}), so this test cannot tell the two apart."
        )
    assert np.array_equal(
        data_alloc.as_numpy(after_turbdiff.zvari(PRESSURE))[columns],
        data_alloc.as_numpy(after_vertdiff.zvari(PRESSURE))[columns],
    ), "ICON's 'zvari(:,:,0)' is not the same at the two exit savepoints after all."

    granule.run_turbdiff(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        dt_tke=entry.dt_tke(),
    )
    between = {
        component: data_alloc.as_numpy(field)[columns].copy() for component, field in components
    }
    circulation = data_alloc.as_numpy(granule._circulation_acceleration)[columns].copy()
    granule.run_vertdiff(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        dt_var=before_vertdiff.dt_var(),
    )

    for component, field in components:
        after = data_alloc.as_numpy(field)[columns]
        gradients = between[component]
        assert not np.array_equal(after, gradients), (
            f"'zvari(:,:,{component})' still holds the vertical gradients 'run_turbdiff' left, "
            "so the second stage did not write the storage the ICON interface shares."
        )
        # Which of the two ICON quantities it ended up as. Stated as a ratio rather than as a
        # tolerance: the agreement itself is gated in 'test_run_reproduces_the_state_icon_-
        # returns', and what is claimed here is only that the distance to one of them is
        # nothing beside the distance to the other.
        right_hand_sides = data_alloc.as_numpy(after_vertdiff.zvari(component))[columns]
        to_the_right_hand_side = np.nanmax(np.abs(after - right_hand_sides))
        to_the_gradient = np.nanmax(np.abs(after - gradients))
        assert to_the_right_hand_side < 1e-6 * to_the_gradient, (
            f"'zvari(:,:,{component})' after both stages is {to_the_right_hand_side} from "
            f"ICON's right-hand side and {to_the_gradient} from the vertical gradient it was "
            "supposed to replace."
        )

    assert np.array_equal(
        data_alloc.as_numpy(granule._circulation_acceleration)[columns], circulation
    ), "'run_vertdiff' wrote 'zvari(:,:,0)', which is not one of its five variables."


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_run_never_writes_the_state_it_was_given(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """ADR-0001 over the pair: the input and surface states come out as they went in.

    The Fortran matches, which is worth checking rather than assuming because it once did not.
    'vertdiff' declares 'u_tens'..'qc_tens' mandatory 'INTENT(INOUT)' (turb_vertdiff.f90:293-302)
    and accumulates into them unconditionally at ':783-806'; the optional in-place incrementation
    of the prognostic variables went upstream in '597f090cf2'. 'u'..'qc' keep 'INTENT(INOUT)'
    only as pointer targets (':451-460'), and ICON adds the tendencies to the state itself at
    'mo_nwp_turbdiff_interface.f90:910-960'. The exit savepoint carries all five prognostic
    variables, so the granule's side of it is measured here and not taken on trust.

    'shfl_s' and 'qvfl_s' are here too. 'vertdiff' would recompute them from the effective
    implicit fluxes at :850-895, but only under '.NOT.(lsfluse .AND. tdc%lsflcnd)', and the
    interface passes 'lsfluse = tdc%lsflcnd' with 'lsflcnd' frozen '.TRUE.'.
    """
    granule, input_state, surface_state, diagnostic, tendency, entry = _states_for_both_stages(
        data_provider, icon_grid, grid_savepoint, date, backend
    )
    before_vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)
    untouched = {
        f"input_state.{name}": data_alloc.as_numpy(getattr(input_state, name)).copy()
        for name in (
            "u",
            "v",
            "w",
            "t",
            "qv",
            "qc",
            "prs",
            "rhoh",
            "epr",
            "tke",
            "ut_sso",
            "vt_sso",
            "tket_conv",
            "hdef2",
            "hdiv",
            "dwdx",
            "dwdy",
        )
    }
    untouched |= {
        f"surface_state.{name}": data_alloc.as_numpy(getattr(surface_state, name)).copy()
        for name in ("t_g", "qv_s", "ps", "l_pat")
    }
    untouched |= {
        f"diagnostic_state.{name}": data_alloc.as_numpy(getattr(diagnostic, name)).copy()
        for name in ("shfl_s", "qvfl_s", "tvm", "tvh")
    }

    granule.run(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        dt_var=before_vertdiff.dt_var(),
        dt_tke=entry.dt_tke(),
    )

    for name, before in untouched.items():
        container, member = name.split(".")
        now = data_alloc.as_numpy(
            getattr(
                {
                    "input_state": input_state,
                    "surface_state": surface_state,
                    "diagnostic_state": diagnostic,
                }[container],
                member,
            )
        )
        assert np.array_equal(before, now), (
            f"'{name}' was written by 'Turbulence.run', which ADR-0001 forbids: "
            f"{np.count_nonzero(before != now)} of {before.size} values changed."
        )


# --------------------------------------- caller fields wider than the grid ---
#
# ICON DOES NOT SIZE ITS FIELDS AT 'grid.num_cells', AND CANNOT BE MADE TO. Every array it hands
# across the py2fgen boundary is a '(:,:,jb)' slice of '(nproma, nlev, nblks)', so its first
# extent is 'nproma'; the granule's own fields are 'grid.num_cells' wide. The two are never
# equal: 'icon4py_init' refuses a configuration with 'nproma < n_patch_edges', and edges always
# outnumber cells on a triangular grid.
#
# Every other test in this package builds both kinds at 'grid.num_cells' -- consistently, which
# is exactly what ICON does not do. That is why 793 passing tests did not stop the first
# verification run inside ICON from dying in "ValueError: operands could not be broadcast
# together with shapes (8320,) (5464,)" at 'turbulence.py:1345', 8320 being 'nproma' and 5464
# 'n_patch_cells' (job 834616, 2026-08-29). So the two tests below build the caller's fields
# DELIBERATELY WIDER, which is the only shape in which the raw-array copies are exercised at all.
#
# The padding is NaN, and that is the second half of the claim: the column window the scheme
# computes is 'ivstart..ivend', well inside 'num_cells', so nothing the granule reads may look at
# those rows and nothing it writes may reach them. A copy that lost its bound either raises or
# poisons an output, and both are failures here.


#: How many undefined columns to put past the grid's last cell. It was 'nproma - n_patch_cells'
#: = 2856 in the run that found this; any positive number exercises the same arithmetic, and a
#: small one keeps the fields small.
PADDING_COLUMNS = 3


def _as_icon_hands_it_over(field: gtx.Field, nproma: int, backend) -> gtx.Field:
    """'field' laid out the way py2fgen hands ICON's memory to the granule.

    Three properties matter and all three are reproduced here.

    * The first extent is 'nproma' and not 'grid.num_cells'. That is the defect's whole shape.
    * The array is Fortran-ordered ('py2fgen/_conversion.py:51'), which is what makes a fixed
      TRAILING index a stride-1 vector and 'ndarray[:num_cells, k]' a contiguous prefix of it --
      the reason the surface-height view can alias the caller's memory instead of copying.
    * The field is built with 'gtx_common._field' over a domain taken from the array's shape,
      which is literally what 'icon4py_export._as_field' does at the boundary.

    The rows past the grid's cells are NaN, or 'False' where the field is a mask: ICON leaves
    that padding undefined and a granule that reads it must not get a plausible number back.
    """
    values = data_alloc.as_numpy(field)
    poison = False if values.dtype == np.bool_ else np.nan
    padded = np.full((nproma, *values.shape[1:]), poison, dtype=values.dtype, order="F")
    padded[: values.shape[0]] = values
    xp = data_alloc.import_array_ns(backend)
    domain = gtx_common.domain(dict(zip(field.domain.dims, padded.shape, strict=True)))
    return gtx_common._field(xp.asarray(padded, order="F"), domain=domain)


def _widened(container, nproma: int, backend):
    """A state container with every field member replaced by its 'nproma'-wide counterpart.

    The containers are frozen dataclasses, so this is 'dataclasses.replace'. Members that are not
    fields -- the empty tracer tuples -- are left alone.
    """
    return dataclasses.replace(
        container,
        **{
            member.name: _as_icon_hands_it_over(getattr(container, member.name), nproma, backend)
            for member in dataclasses.fields(container)
            if isinstance(getattr(container, member.name), gtx.Field)
        },
    )


def _agrees(computed: np.ndarray, reference: np.ndarray) -> bool:
    """Bit-for-bit, with NaN counting as equal to NaN.

    The narrow run starts 'rhon' and the updated TKE at NaN on purpose, and the rows neither
    stage writes stay NaN in both runs. Those rows are the interesting ones here, so they are
    compared rather than excluded.
    """
    return bool(np.array_equal(computed, reference, equal_nan=computed.dtype.kind == "f"))


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES[:1])
def test_the_granule_takes_metric_fields_wider_than_the_grid(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """Construction survives 'nproma > num_cells', and the surface height ALIASES ICON's array.

    This is the traceback of job 834616 turned into a test: building the working set is where the
    granule first reads a caller-supplied array by hand, and it did it with no bound at all.

    The fix is the dycore's ('solve_nonhydro.py:885-899'): GT4Py cannot slice a vertical level out
    of a field, so 'hhl(:,ke1)' is wrapped as a one-dimensional view with 'gtx_common._field'
    rather than copied into a grid-sized buffer. The aliasing is asserted twice -- once through
    'shares_memory' and once by writing into ICON's array and reading the value back out of the
    view -- because it is not decoration. 'dp0' is bound by the same metric state and aliases
    'p_diag%dpres_mc', which ICON recomputes every timestep; a copy anywhere on this path would
    freeze a field at step 0 and drift like a physics bug rather than fail like a defect.

    No stencil runs, so this test is not marked 'uses_concat_where' and is the one of the pair
    that reports on 'embedded'.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    nlev = int(entry.ke())
    num_cells = icon_grid.num_cells
    nproma = num_cells + PADDING_COLUMNS
    metric_state = _widened(
        states.TurbulenceMetricState(
            hhl=utils.copy_of(entry.hhl(), backend),
            dp0=utils.copy_of(entry.dp0(), backend),
            l_hori=utils.copy_of(entry.l_hori(), backend),
            trop_mask=utils.copy_of(entry.trop_mask(), backend),
            innertrop_mask=utils.copy_of(entry.innertrop_mask(), backend),
        ),
        nproma,
        backend,
    )
    assert metric_state.hhl.ndarray.shape[0] == nproma, "the caller's field was not widened"

    granule = turbulence.Turbulence(
        grid=icon_grid,
        config=CONFIG,
        params=PARAMS,
        vertical_grid=v_grid.VerticalGrid(
            config=v_grid.VerticalGridConfig(num_levels=nlev),
            vct_a=grid_savepoint.vct_a(),
            vct_b=grid_savepoint.vct_b(),
        ),
        metric_state=metric_state,
        backend=backend,
    )

    surface_height = granule._surface_height
    assert surface_height.ndarray.shape == (num_cells,), (
        "'_surface_height' is the granule's own field and must be the grid's width, not "
        f"'nproma': got {surface_height.ndarray.shape}."
    )
    assert _agrees(
        data_alloc.as_numpy(surface_height), data_alloc.as_numpy(metric_state.hhl)[:num_cells, nlev]
    ), "'_surface_height' is not 'hhl(:,ke1)' over the grid's cells."

    xp = data_alloc.import_array_ns(backend)
    assert xp.shares_memory(surface_height.ndarray, metric_state.hhl.ndarray), (
        "'_surface_height' no longer aliases the caller's 'hhl'. It has to be a view: the same "
        "metric state binds 'dp0', which aliases ICON's 'p_diag%dpres_mc' and is recomputed "
        "every timestep, so a copy on this path would silently use step 0's values forever."
    )
    # The same claim behaviourally, which is what actually matters and holds on every backend.
    poked = -12345.0
    metric_state.hhl.ndarray[num_cells - 1, nlev] = poked
    assert float(data_alloc.as_numpy(surface_height)[num_cells - 1]) == poked, (
        "a write into the caller's 'hhl' did not show through '_surface_height', so the two are "
        "separate buffers."
    )

    # 'l_hori' is read by hand too, and the two scales derived from it are genuinely new data --
    # so they are written, not viewed, and the read is what has to be clamped. If it were not,
    # the NaN padding would land in both of them.
    for name in ("_horizontal_length_scale_limit", "_minimal_tke_forcing"):
        derived = data_alloc.as_numpy(getattr(granule, name))
        assert derived.shape == (num_cells,), f"'{name}' has the wrong width: {derived.shape}."
        assert np.isfinite(derived).all(), (
            f"'{name}' picked up ICON's undefined padding: {np.count_nonzero(~np.isfinite(derived))}"
            f" of {derived.size} values are not finite."
        )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES[:1])
def test_run_takes_caller_fields_wider_than_the_grid(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """The pair runs on ICON-shaped fields and computes exactly what it computes on grid-shaped
    ones.

    The same states twice, once at 'grid.num_cells' and once at 'nproma', compared bit for bit
    over the grid's cells. No gate and no tolerance: padding is not arithmetic, so anything but
    identity is a defect. That covers the four raw-array copies inside 'run_turbdiff' --
    '_extract_level' on 'tkvm' and 'tkvh', '_set_level' on 'ps', '_copy_level' on 'tke' and
    '_copy_levels' on 'rcld' -- each of which has a caller-supplied field on one side and one of
    the granule's own on the other.

    The padding rows are checked separately, and they are the half of this that a same-answer
    comparison cannot see: an unclamped write would put the granule's numbers into memory ICON
    considers undefined, and every output would still be right.
    """
    narrow = _run(data_provider, icon_grid, grid_savepoint, date, backend)

    granule, input_state, surface_state, diagnostic, tendency, entry = _states_for_both_stages(
        data_provider, icon_grid, grid_savepoint, date, backend
    )
    num_cells = icon_grid.num_cells
    nproma = num_cells + PADDING_COLUMNS
    metric_state = _widened(granule._metric_state, nproma, backend)
    input_state = _widened(input_state, nproma, backend)
    surface_state = _widened(surface_state, nproma, backend)
    diagnostic = _widened(diagnostic, nproma, backend)
    tendency = _widened(tendency, nproma, backend)
    before_vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)

    turbulence.Turbulence(
        grid=icon_grid,
        config=CONFIG,
        params=PARAMS,
        vertical_grid=v_grid.VerticalGrid(
            config=v_grid.VerticalGridConfig(num_levels=int(entry.ke())),
            vct_a=grid_savepoint.vct_a(),
            vct_b=grid_savepoint.vct_b(),
        ),
        metric_state=metric_state,
        backend=backend,
    ).run(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic,
        tendency_state=tendency,
        dt_var=before_vertdiff.dt_var(),
        dt_tke=entry.dt_tke(),
    )

    compared = 0
    for container, wide in (
        ("diagnostic_state", diagnostic),
        ("tendency_state", tendency),
    ):
        grid_shaped = {
            "diagnostic_state": narrow.diagnostic_state,
            "tendency_state": narrow.tendency_state,
        }[container]
        for member in dataclasses.fields(wide):
            got = getattr(wide, member.name)
            if not isinstance(got, gtx.Field):
                continue
            compared += 1
            on_nproma = data_alloc.as_numpy(got)
            on_the_grid = data_alloc.as_numpy(getattr(grid_shaped, member.name))
            assert _agrees(on_nproma[:num_cells], on_the_grid), (
                f"'{container}.{member.name}' differs between the 'nproma'-wide run and the "
                "'num_cells'-wide one, so a raw-array copy is reading or writing the wrong "
                "columns."
            )
            if on_nproma.dtype.kind != "f":
                continue
            assert np.isnan(on_nproma[num_cells:]).all(), (
                f"'{container}.{member.name}' was written past the grid's last cell: "
                f"{np.count_nonzero(~np.isnan(on_nproma[num_cells:]))} of "
                f"{on_nproma[num_cells:].size} padding values are no longer undefined."
            )
    assert compared > 20, f"only {compared} outputs were compared; the containers changed shape."

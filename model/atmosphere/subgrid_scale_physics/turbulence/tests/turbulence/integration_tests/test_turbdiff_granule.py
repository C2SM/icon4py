# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end datatest of 'Turbulence.run_turbdiff': 'turbdiff-entry' to 'turbdiff-exit'.

The nine section datatests each start from their own entry savepoint, so each one verifies its
section in isolation and none of them can see a composition error. This module runs the whole of
'SUBROUTINE turbdiff' from the state ICON handed it and compares what comes out against the
state ICON handed back, for all four timesteps of exp.mch_icon-ch2_small. It is the only test in
the package that can fail on an ordering mistake, a storage-aliasing mistake or a vertical
domain that the section it belongs to happens not to read.

WHAT ONLY THIS TEST CAN CATCH
-----------------------------
* Section 8) and section 9) are a PAIR. Without the circulation term neither the virtual TKE
  profile nor its subtraction may run, and no stencil can enforce that; the driver has to.
* 'len_scale' is the turbulent master length scale until section 6) reads it for the last time
  and the right-hand side of the TKE solve from section 9) on. Eleven more storages are reused
  the same way -- 'frh', 'frm', 'hlp', 'dicke', 'rcld', 'zvari(:,:,0..5)' and 'zaux(:,:,1..5)'.
  Every section respects the reuse locally; nothing else checks the handover.
* 'mean_shear_forcing' is an intermediate of section 2a) that no savepoint holds. The section
  test allocates it; the granule has to.
* Which outputs are accumulated into and which are overwritten. 'ddt_tke' arrives holding the
  advection tendency, is read as such by section 3), and is overwritten by section 10).

WHY THIS XFAILS ON 'embedded'
-----------------------------
The chain contains five programs that select a boundary row with 'concat_where', which gt4py
1.1.10 cannot execute on the embedded backend (package README, "Boundary rows"), so the whole
run is marked 'uses_concat_where' and aborts there before it starts. Embedded coverage of the
sections themselves is unaffected -- each section datatest keeps whichever of its programs are
'concat_where'-free -- but the composition is validated on the compiled backends only. That the
mark also spares embedded the two 'scan_operator's, which it executes as Python loops at minutes
per date, is a convenience and not the reason.

THE CONFIGURATION IS THE EXPERIMENT'S, NOT THE COMPILED-IN DEFAULT
------------------------------------------------------------------
'&turbdiff_nml' of 'icon/run/exp.mch_icon-ch2_small' sets seventeen parameters and four of them
change what this test computes: 'a_hshr = 2.0' against a default of 1.0 (section 2a),
'q_crit = 2.0' against 1.6 (section 0), 'tkhmin = 0.5' against 0.75 (section 4) and
'tur_len = 300' against 500 (the length-scale limit 'l_scal', section 0). 'CONFIG' below is that
namelist; 'test_the_configuration_is_the_one_the_capture_ran' recovers the last of them from the
data rather than trusting the file.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence,
    turbulence_states as states,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: '&turbdiff_nml' of 'icon/run/exp.mch_icon-ch2_small', entry for entry. The three parameters
#: it sets that this granule does not read ('rat_sea', 'rlam_heat', 'alpha1', 'imode_charpar')
#: belong to 'turbtran' and are carried anyway, so that the object is the experiment's namelist
#: and not a subset somebody chose.
CONFIG = turbulence.TurbulenceConfig(
    tkhmin=0.5,
    tkmmin=0.75,
    pat_len=750.0,
    tur_len=300.0,
    rat_sea=0.8,
    ltkesso=True,
    frcsmot=0.0,
    imode_frcsmot=2,
    itype_sher=2,
    ltkeshs=True,
    a_hshr=2.0,
    icldm_turb=2,
    q_crit=2.0,
    imode_tkesso=2,
    rlam_heat=10.0,
    alpha1=0.125,
    imode_charpar=3,
)
PARAMS = turbulence.TurbulenceParams(CONFIG)

#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), zero-based as the reader takes them.
PRESSURE, U_M, V_M, TET_L, H2O_G, LIQ = 0, 1, 2, 3, 4, 5


#: The outputs that come out of the whole chain BIT-EXACT, measured on 'gtfn_cpu' and
#: 'dace_cpu' at all four dates, each with the reason it escapes the two transcendentals.
#:
#: Those two are the only inexact steps in the whole of 'turbdiff': section 0)'s Magnus formula
#: for the saturation vapour pressure and its Exner factor ('EXP' and 'EXP(LOG())'), and section
#: 2a)'s 'xri = EXP(2/3*LOG(frm/frh))'. Every other section is gated 'Exact()' in
#: 'gate_registry'. So a quantity that descends from neither survives forty-two programs with
#: ICON's bits, and that is what these four assert.
BIT_EXACT_END_TO_END: tuple[tuple[str, str], ...] = (
    (
        "zvari(:,:,u_m)",
        "The zonal wind reaches the gradient through 'u', 'tfm', 'hhl' and 'rhon' on the "
        "interpolated rows, and no exponential is evaluated anywhere along that path.",
    ),
    ("zvari(:,:,v_m)", "As the zonal wind."),
    (
        "zvari(:,:,h2o_g)",
        "The total water is 'qv + qc'; 'adjust_satur_equil' forms it without the saturation "
        "vapour pressure, which is what section 0)'s "
        "'test_the_conserved_variables_are_bit_exact_where_no_exponential_is_involved' "
        "measures one section at a time.",
    ),
    (
        "dicke",
        "The layer depth is a difference of 'hhl'; the discretisation momentum that replaces "
        "it multiplies that by the half-level density, which on the rows section 1a) writes is "
        "the mass-weighted interpolation of ICON's own 'rhoh'.",
    ),
)


#: Gate key for every output of the stage but one. There is no single stencil to key an
#: end-to-end comparison on, so the granule declares its own entries; a diff of them is still
#: the record of where the numerics moved, which is what 'gate_registry' exists for.
GRANULE_GATE = "run_turbdiff"

#: Gate key for 'tketens' alone, which needs two decades more than everything else.
#:
#: Not because its own arithmetic is worse -- section 10) has no transcendental and no
#: multiply-add exposure, and its stencil is gated 'Exact()'. It is a DIFFERENCE OF TWO NEARLY
#: EQUAL NUMBERS: 'tketens = (q_new - q_old)*fr_tke' with the two profiles agreeing to five or
#: six digits, so an absolute error the size of an ULP of 'q' becomes a relative error the size
#: of an ULP of 'q' divided by 'q_new - q_old'. The measurement shows exactly that shape -- the
#: largest RELATIVE error is 4.0e-9 while the largest ABSOLUTE error over the same values is
#: 1.8e-15, which is one ULP of a quantity of order 1. Giving it the same gate as the rest would
#: mean widening the rest by two decades for a reason that does not apply to them.
TKE_TENDENCY_GATE = "run_turbdiff_tke_tendency"


def _gate_for(quantity: str) -> str:
    """Which of the two granule gates an output is compared under."""
    return TKE_TENDENCY_GATE if quantity == "tketens" else GRANULE_GATE


class Turbdiff(NamedTuple):
    """One timestep run end to end: the granule, the states it wrote and the reference."""

    granule: turbulence.Turbulence
    entry: sb.IconTurbdiffEntrySavepoint
    after: sb.IconTurbulenceSavepoint
    diagnostic_state: states.TurbulenceDiagnosticState
    tendency_state: states.TurbulenceTendencyState
    nlev: int
    #: Half-open range of columns 'turbdiff' computed; every comparison below is masked with it.
    columns: slice


def _unused_cell_field(grid, backend) -> gtx.Field:
    """A cell field for a state member 'turbdiff' never reads."""
    return data_alloc.zero_field(grid, dims.CellDim, allocator=backend)


def _unused_cell_k_field(grid, backend, *, half: bool = True) -> gtx.Field:
    """A (Cell, K) field for a state member 'turbdiff' never reads."""
    extend = {dims.KDim: 1} if half else None
    return data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend=extend, allocator=backend)


def _run_turbdiff(data_provider, icon_grid, grid_savepoint, date: str, backend) -> Turbdiff:
    """Build the granule's states from 'turbdiff-entry' and run the whole stage once."""
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    section_0 = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_exit(date=date)
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
        # 'rhon' and the updated TKE are pure outputs of 'turbdiff' with no entry state, so
        # they are NaN-filled: a row the granule fails to write must not hold a plausible value.
        rhon=utils.nan_like(entry.rcld(), backend),
        updated_tke=utils.nan_like(entry.tke(), backend),
        # Written by 'turbtran', never by 'turbdiff'.
        tcm=_unused_cell_field(icon_grid, backend),
        tch=_unused_cell_field(icon_grid, backend),
        shfl_s=_unused_cell_field(icon_grid, backend),
        qvfl_s=_unused_cell_field(icon_grid, backend),
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
        # 'tketens' is INTENT(INOUT): the advection tendency on the way in, which section 3)
        # reads as 'tvt', and the diffusion tendency on the way out.
        ddt_tke=utils.copy_of(entry.tketens(), backend),
        # 'tket_hshr' is not serialized at 'turbdiff-entry' -- it is an output slot the entry
        # hook does not carry -- so its entry state is read from 'turbdiff-0-exit', which
        # sections 0) and 1) leave untouched. The rows section 2a) does not write are then
        # asserted unchanged like every other output.
        tket_hshr=utils.copy_of_raw_field(section_0, "td_tket_hshr", backend),
        # Written by 'vertdiff' only.
        ddt_u=_unused_cell_k_field(icon_grid, backend, half=False),
        ddt_v=_unused_cell_k_field(icon_grid, backend, half=False),
        ddt_t=_unused_cell_k_field(icon_grid, backend, half=False),
        ddt_qv=_unused_cell_k_field(icon_grid, backend, half=False),
        ddt_qc=_unused_cell_k_field(icon_grid, backend, half=False),
        ddt_tracers=(),
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
    granule.run_turbdiff(
        input_state=input_state,
        surface_state=surface_state,
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        dt_tke=entry.dt_tke(),
    )
    return Turbdiff(
        granule=granule,
        entry=entry,
        after=after,
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        nlev=nlev,
        columns=slice(entry.ivstart(), entry.ivend()),
    )


def _outputs(run: Turbdiff) -> tuple[tuple[str, gtx.Field, gtx.Field, slice], ...]:
    """Every quantity 'turbdiff' returns, with the rows ICON defines for it.

    A row window narrower than the whole column appears only where ICON itself leaves the row
    undefined, and each one says which.
    """
    nlev, granule = run.nlev, run.granule
    diagnostic, tendency, after = run.diagnostic_state, run.tendency_state, run.after
    everything = slice(0, nlev + 1)
    return (
        # -- the interface fields --------------------------------------------------------------
        ("tke", diagnostic.updated_tke, after.tke(), everything),
        ("tkvm", diagnostic.tkvm, after.tkvm(), everything),
        ("tkvh", diagnostic.tkvh, after.tkvh(), everything),
        ("rcld", diagnostic.rcld, after.rcld(), everything),
        # 'rhon(:,1)' is never written by 'turbdiff': 'bound_level_interp' starts at k=2 and
        # 'adjust_satur_equil' supplies 'ke1', so the model top holds untouched memory (around
        # -0.025 in this capture) on ICON's side of the comparison.
        ("rhon", diagnostic.rhon, after.rhon(), slice(1, nlev + 1)),
        ("tketens", tendency.ddt_tke, after.tketens(), everything),
        ("tket_hshr", tendency.tket_hshr, after.tket_hshr(), everything),
        # -- 'zvari', which 'vertdiff' receives as 'vd_zvari_in' ---------------------------------
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
        # Component 0 is the half-level pressure until section 3) replaces it with the
        # circulation acceleration over rows 1..ke1; the model top is never written.
        (
            "zvari(:,:,0) [circulation acceleration]",
            granule._circulation_acceleration,
            after.zvari(PRESSURE),
            slice(1, nlev + 1),
        ),
        # -- the routine locals the exit hook carries for localisation ----------------------------
        ("frh", granule._frh, after.frh(), slice(1, nlev + 1)),
        ("frm", granule._frm, after.frm(), slice(1, nlev + 1)),
        ("hlp", granule._hlp, after.hlp(), everything),
        # 'dicke(:,ke1)' is untouched memory: section 0) writes main levels 1..ke and section
        # 1a) half levels 2..ke.
        ("dicke", granule._dicke, after.dicke(), slice(0, nlev)),
        ("len_scale", granule._len_scale, after.len_scale(), everything),
        # 'zaux(:,1,1)' is untouched memory for the same reason 'rhon(:,1)' is.
        ("zaux(:,:,1)", granule._zaux_1, after.zaux(0), slice(1, nlev + 1)),
        ("zaux(:,:,2)", granule._zaux_2, after.zaux(1), everything),
        ("zaux(:,:,3)", granule._zaux_3, after.zaux(2), everything),
        ("zaux(:,:,4)", granule._zaux_4, after.zaux(3), everything),
        ("zaux(:,:,5)", granule._zaux_5, after.zaux(4), everything),
    )


# ------------------------------------------------------------------ what the capture ran ---


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_configuration_is_the_one_the_capture_ran(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'CONFIG' is 'exp.mch_icon-ch2_small's namelist, recovered from the data where it can be.

    Two of 'turb_setup's outputs are serialized and both are pure functions of 'l_hori' and the
    configuration, so they pin 'tur_len' and 'vel_min' exactly -- 'tur_len = 300' is the value
    the granule's turbulent length scale depends on and it is NOT the compiled-in 500. The
    third assertion pins the surface values the second 'adjust_satur_equil' call starts from,
    which the granule takes from 'TurbulenceSurfaceState' rather than from a 'zvari' row.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    columns = slice(entry.ivstart(), entry.ivend())
    l_hori = entry.l_hori().asnumpy()

    length_scale_limit = np.minimum(0.5 * l_hori, CONFIG.tur_len)
    velocity_scale = CONFIG.vel_min / np.maximum(l_hori, CONFIG.tur_len)
    assert np.array_equal(length_scale_limit[columns], entry.l_scal().asnumpy()[columns]), (
        "'l_scal' does not follow from 'tur_len'; the capture ran a different namelist."
    )
    assert np.array_equal(
        (velocity_scale * velocity_scale)[columns], entry.fc_min().asnumpy()[columns]
    ), "'fc_min' does not follow from 'vel_min' and 'tur_len'."

    surface = data_alloc.as_numpy(entry.zvari(PRESSURE))[columns, entry.ke()]
    assert np.array_equal(surface, entry.ps().asnumpy()[columns])
    assert np.array_equal(
        data_alloc.as_numpy(entry.zvari(TET_L))[columns, entry.ke()],
        entry.t_g().asnumpy()[columns],
    )
    assert np.array_equal(
        data_alloc.as_numpy(entry.zvari(H2O_G))[columns, entry.ke()],
        entry.qv_s().asnumpy()[columns],
    )
    assert not np.any(data_alloc.as_numpy(entry.zvari(LIQ))[columns, entry.ke()]), (
        "the surface liquid water is not zero, so 'ilow_def_cond' is not 2."
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
def test_the_computed_columns_are_the_grids_prognostic_cells(
    *, data_provider: sb.IconSerialDataProvider, icon_grid
) -> None:
    """'_determine_horizontal_domains' reproduces the interface's 'rl_start'/'rl_end'.

    'mo_nwp_turbdiff_interface.f90' calls with 'grf_bdywidth_c + 1' to 'min_rlcell_int'. The
    granule has no 'ivstart' to read, so it derives the window from the grid; that the two agree
    is what makes every comparison in this module meaningful.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=utils.TURBDIFF_DATES[0])
    cell_domain = h_grid.domain(dims.CellDim)
    assert int(icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))) == entry.ivstart()
    assert int(icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))) == entry.ivend()


# ------------------------------------------------------------------------- the whole stage ---


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_run_turbdiff_reproduces_the_state_icon_returns(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """Every quantity 'turbdiff' returns agrees with ICON under the granule's gate.

    The comparison covers the whole column of each output, not only the rows the section that
    wrote it is responsible for, so a section whose vertical domain is one row wide of the
    Fortran's fails here even where its own datatest cannot see it.

    This needs 'GRANULE_GATE' and 'TKE_TENDENCY_GATE' in 'gate_registry.GATES'; without them
    'gate_for' raises 'UnregisteredStencilError' rather than defaulting to 'Exact()', which is
    the registry working as intended. What it must NOT hide is the four outputs that are
    bit-exact end to end: those are asserted ungated by the test below, so widening either gate
    cannot quietly absorb a new disagreement in them.
    """
    run = _run_turbdiff(data_provider, icon_grid, grid_savepoint, date, backend)
    for quantity, computed, reference, levels in _outputs(run):
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
def test_the_quantities_that_survive_the_chain_bit_exactly(
    date: str, *, data_provider: sb.IconSerialDataProvider, icon_grid, grid_savepoint, backend
) -> None:
    """The outputs that reach ICON's numbers exactly, asserted without a gate.

    A single end-to-end tolerance would hide this: section 0)'s 'EXP' and section 2a)'s
    'EXP(LOG())' are the only inexact steps of the whole routine, and a quantity that does not
    descend from either is bit-exact after all forty-two programs. Listing which ones those are
    is worth more than the tolerance, and asserting it ungated is what keeps the list honest --
    widening the gate cannot quietly absorb a new disagreement here.
    """
    run = _run_turbdiff(data_provider, icon_grid, grid_savepoint, date, backend)
    exact = dict(BIT_EXACT_END_TO_END)
    for quantity, computed, reference, levels in _outputs(run):
        if quantity not in exact:
            continue
        got = data_alloc.as_numpy(computed)[run.columns, levels]
        want = data_alloc.as_numpy(reference)[run.columns, levels]
        assert np.array_equal(got, want), (
            f"'{quantity}' was bit-exact end to end when the gate was measured and is not any "
            f"more: max abs {np.nanmax(np.abs(got - want))} over "
            f"{np.count_nonzero(got != want)} of {got.size} values. {exact[quantity]}"
        )

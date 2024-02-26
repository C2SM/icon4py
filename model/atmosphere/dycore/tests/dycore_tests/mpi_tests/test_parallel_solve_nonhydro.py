# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    PrepAdvection,
)
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa : F401 fixture
    decomposition_info,
)
from icon4py.model.common.test_utils.helpers import dallclose, zero_field
from icon4py.model.common.test_utils.parallel_helpers import (  # noqa : F401 fixture
    check_comm_size,
    processor_props,
)

from ..test_solve_nonhydro import create_prognostic_states
from ..utils import (
    construct_config,
    construct_interpolation_state_for_nonhydro,
    construct_nh_metric_state,
)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, jstep_init, step_date_init,istep_exit, jstep_exit, step_date_exit",
    [(1, 0, "2021-06-20T12:00:10.000", 2, 0, "2021-06-20T12:00:10.000")],
)
@pytest.mark.mpi
def test_run_solve_nonhydro_single_step(
    istep_init,
    istep_exit,
    jstep_init,
    jstep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    icon_grid,
    savepoint_nonhydro_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
    processor_props,  # noqa : F811 fixture
    decomposition_info,  # noqa : F811 fixture
):
    check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing dycore for experiment 'mch_ch_r04_b09_dsl"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels} "
        f"local cells = {decomposition_info.global_index(CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    owned_cells = decomposition_info.owner_mask(CellDim)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo edges {np.count_nonzero(np.invert( decomposition_info.owner_mask(EdgeDim)))}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )

    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=grid_savepoint.nflat_gradp(),
        nflatlev=grid_savepoint.nflatlev(),
    )
    sp_v = savepoint_velocity_init
    dtime = sp_v.get_metadata("dtime").get("dtime")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=zero_field(icon_grid, CellDim, KDim),
    )

    nnow = 0
    nnew = 1
    recompute = sp_v.get_metadata("recompute").get("recompute")
    linit = sp_v.get_metadata("linit").get("linit")

    diagnostic_state_nh = DiagnosticStateNonHydro(
        theta_v_ic=sp.theta_v_ic(),
        exner_pr=sp.exner_pr(),
        rho_ic=sp.rho_ic(),
        ddt_exner_phy=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_fl_e=sp.mass_fl_e(),
        ddt_vn_phy=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=sp.exner_dyn_incr(),
    )
    initial_divdamp_fac = sp.divdamp_fac_o2()
    interpolation_state = construct_interpolation_state_for_nonhydro(interpolation_savepoint)
    metric_state_nonhydro = construct_nh_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_state_ls = create_prognostic_states(sp)
    prognostic_state_nnew = prognostic_state_ls[1]

    exchange = definitions.create_exchange(processor_props, decomposition_info)

    solve_nonhydro = SolveNonhydro(exchange)
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  entering : solve_nonhydro.time_step"
    )

    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_ls=prognostic_state_ls,
        prep_adv=prep_adv,
        divdamp_fac_o2=initial_divdamp_fac,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
        at_first_substep=jstep_init == 0,
        at_last_substep=jstep_init == (ndyn_substeps - 1),
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: dycore step run ")

    expected_theta_v = sp_step_exit.theta_v_new().asnumpy()
    calculated_theta_v = prognostic_state_nnew.theta_v.asnumpy()
    assert dallclose(
        expected_theta_v,
        calculated_theta_v,
    )
    expected_exner = sp_step_exit.exner_new().asnumpy()
    calculated_exner = prognostic_state_nnew.exner.asnumpy()
    assert dallclose(
        expected_exner,
        calculated_exner,
    )
    assert dallclose(
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        prognostic_state_nnew.vn.asnumpy(),
        rtol=1e-10,
    )
    assert dallclose(
        savepoint_nonhydro_exit.w_new().asnumpy(),
        prognostic_state_nnew.w.asnumpy(),
        atol=8e-14,
    )
    assert dallclose(
        savepoint_nonhydro_exit.rho_new().asnumpy(),
        prognostic_state_nnew.rho.asnumpy(),
    )

    assert dallclose(
        savepoint_nonhydro_exit.rho_ic().asnumpy(),
        diagnostic_state_nh.rho_ic.asnumpy(),
    )

    assert dallclose(
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
        diagnostic_state_nh.theta_v_ic.asnumpy(),
    )

    assert dallclose(
        savepoint_nonhydro_exit.mass_fl_e().asnumpy(),
        diagnostic_state_nh.mass_fl_e.asnumpy(),
        rtol=1e-10,
    )

    assert dallclose(
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        prep_adv.mass_flx_me.asnumpy(),
        rtol=1e-10,
    )
    assert dallclose(
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        prep_adv.vn_traj.asnumpy(),
        rtol=1e-10,
    )

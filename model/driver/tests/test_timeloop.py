import pytest
import numpy as np

from icon4py.model.driver.dycore_driver import TimeLoop

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.prep_adv_state import PrepAdvection
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams

from icon4py.model.common.grid.horizontal import EdgeParams, CellParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config

from icon4py.model.common.test_utils.helpers import dallclose, random_field, zero_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh




# step_date_init and step_date_exit are used in diffusion, declared in fixtures
# testing on MCH_CH_r04b09_dsl data
# TODO include timeloop_date_start, = "2021-06-20T12:00:00.000", for icon_run_config
@pytest.mark.datatest
@pytest.mark.parametrize(
    "timeloop_date_init, timeloop_date_exit",
    [("2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_run_timeloop_single_step(
    timeloop_date_init,
    timeloop_date_exit,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    interpolation_savepoint,
    r04b09_diffusion_config,
    r04b09_iconrun_config,
    damping_height,
    timeloop_diffusion_savepoint_init,
    timeloop_diffusion_savepoint_exit,
    timeloop_nonhydro_savepoint_init,
    timeloop_velocity_savepoint_init,
    timeloop_nonhydro_savepoint_exit,
    timeloop_nonhydro_step_savepoint_exit,
):
    diffusion_config = r04b09_diffusion_config
    diffusion_dtime = timeloop_diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    diffusion_interpolation_state = interpolation_savepoint.construct_interpolation_state_for_diffusion()
    diffusion_metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    diffusion_diagnostic_state = timeloop_diffusion_savepoint_init.construct_diagnostics_for_diffusion()
    prognostic_state = timeloop_diffusion_savepoint_init.construct_prognostics()
    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=grid_savepoint.nflatlev(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
    )
    additional_parameters = DiffusionParams(diffusion_config)

    #verify_diffusion_fields(diffusion_diagnostic_state, before_dycore_prognostic_state, diffusion_savepoint_init)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=diffusion_config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    '''
    verify_diffusion_fields(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
    '''

    # TODO a wrapper for constructing explicitly the MCH_CH_r04b09_dsl run config for nonhydro
    nonhydro_config = NonHydrostaticConfig()
    sp = timeloop_nonhydro_savepoint_init #savepoint_nonhydro_init
    sp_step_exit = timeloop_nonhydro_step_savepoint_exit
    nonhydro_params = NonHydrostaticParams(nonhydro_config)
    #sp_d = data_provider.from_savepoint_grid()
    sp_v = timeloop_velocity_savepoint_init
    mesh = SimpleMesh()
    nonhydro_dtime = timeloop_velocity_savepoint_init.get_metadata("dtime").get("dtime")
    # TODO lprep_adv actually depends on other factors: idiv_method == 1 .AND. (ltransport .OR. p_patch%n_childdom > 0 .AND. grf_intmethod_e >= 5)
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    #clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx") # moved to time loop
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    #recompute = sp_v.get_metadata("recompute").get("recompute") # moved to time loop
    #linit = sp_v.get_metadata("linit").get("linit") # moved to time loop
    #dyn_timestep = sp_v.get_metadata("dyn_timestep").get("dyn_timestep")


    assert timeloop_diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v
    assert r04b09_iconrun_config.dtime == diffusion_dtime

    z_fields = ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_beta=_allocate(CellDim, KDim, mesh=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, mesh=icon_grid),
        z_q=_allocate(CellDim, KDim, mesh=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, mesh=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, mesh=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, mesh=icon_grid),
    )

    nh_constants = NHConstants(
        wgt_nnow_rth=sp.wgt_nnow_rth(),
        wgt_nnew_rth=sp.wgt_nnew_rth(),
        wgt_nnow_vel=sp.wgt_nnow_vel(),
        wgt_nnew_vel=sp.wgt_nnew_vel(),
        scal_divdamp=sp.scal_divdamp(),
        scal_divdamp_o2=sp.scal_divdamp_o2(),
    )

    nonhydro_interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    nonhydro_metric_state = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=nonhydro_metric_state,
        interpolation_state=nonhydro_interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_areas=cell_geometry.area,
        owner_mask=grid_savepoint.c_owner_mask(),
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        fac=fac,
        z=z,
    )

    nonhydro_diagnostic_state = DiagnosticStateNonHydro(
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
    )

    timeloop = TimeLoop(
        r04b09_iconrun_config,
        diffusion,
        solve_nonhydro
    )

    assert timeloop.substep_timestep == nonhydro_dtime

    timeloop.time_integration(
        diffusion_diagnostic_state,
        nonhydro_diagnostic_state,
        prognostic_state,
        prep_adv,
        z_fields,
        nh_constants,
        sp.bdy_divdamp(),
        lprep_adv
    )

    try:
        assert np.allclose(
            np.asarray(timeloop_diffusion_savepoint_exit.vn()),
            np.asarray(prognostic_state.vn)
        )
    except:
        print("vn is not the same")
        print( np.max( np.abs( np.asarray(timeloop_diffusion_savepoint_exit.vn()) - np.asarray(prognostic_state.vn) ) ) )
        print(np.max(np.abs(np.asarray(timeloop_diffusion_savepoint_exit.vn()))))
        print(np.max(np.abs(np.asarray(prognostic_state.vn))))

    try:
        assert np.allclose(
            np.asarray(timeloop_diffusion_savepoint_exit.w()),
            np.asarray(prognostic_state.w)
        )
    except:
        print("w is not the same")
        print( np.max( np.abs( np.asarray(timeloop_diffusion_savepoint_exit.w()) - np.asarray(prognostic_state.w) ) ) )
        print(np.max(np.abs(np.asarray(timeloop_diffusion_savepoint_exit.w()))))
        print(np.max(np.abs(np.asarray(prognostic_state.w))))

    try:
        assert np.allclose(
            np.asarray(timeloop_diffusion_savepoint_exit.exner()),
            np.asarray(prognostic_state.exner)
        )
    except:
        print("exner is not the same")
        print( np.max( np.abs( np.asarray(timeloop_diffusion_savepoint_exit.exner()) - np.asarray(prognostic_state.exner) ) ) )
        print(np.max(np.abs(np.asarray(timeloop_diffusion_savepoint_exit.exner()))))
        print(np.max(np.abs(np.asarray(prognostic_state.exner))))

    try:
        assert np.allclose(
            np.asarray(timeloop_diffusion_savepoint_exit.theta_v()),
            np.asarray(prognostic_state.theta_v)
        )
    except:
        print("thehta_v is not the same")
        print( np.max( np.abs( np.asarray(timeloop_diffusion_savepoint_exit.theta_v()) - np.asarray(prognostic_state.theta_v) ) ) )
        print(np.max(np.abs(np.asarray(timeloop_diffusion_savepoint_exit.theta_v()))))
        print(np.max(np.abs(np.asarray(prognostic_state.theta_v))))

    try:
        assert np.allclose(
            np.asarray(timeloop_nonhydro_savepoint_exit.rho()),
            np.asarray(prognostic_state.rho)
        )
    except:
        print("rho is not the same")
        print( np.max( np.abs( np.asarray(timeloop_nonhydro_savepoint_exit.rho()) - np.asarray(prognostic_state.rho) ) ) )
        print(np.max(np.abs(np.asarray(timeloop_nonhydro_savepoint_exit.rho()))))
        print(np.max(np.abs(np.asarray(prognostic_state.rho))))

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

from icon4py.nh_solve.solve_nonydro import NonHydrostaticParams, SolveNonhydro
from icon4py.state_utils.diagnostic_state import (
    DiagnosticState,
    DiagnosticStateNonHydro,
)
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.z_fields import ZFields
from icon4py.velocity.velocity_advection import VelocityAdvection


@pytest.mark.datatest
def test_nonhydro_params():
    nonhydro_params = NonHydrostaticParams(config)

    assert nonhydro_params.df32 == pytest.approx(nonhydro_params.divdamp_fac3 - nonhydro_params.divdamp_fac2, abs=1e-12)
    assert nonhydro_params.dz32 == pytest.approxnonhydro_params.divdamp_z3 - nonhydro_params.divdamp_z2, abs=1e-12)
    assert nonhydro_params.df42 == pytest.approxnonhydro_params.divdamp_fac4 - nonhydro_params.divdamp_fac2, abs=1e-12)
    assert nonhydro_params.dz42 == pytest.approxnonhydro_params.divdamp_z4 - nonhydro_params.divdamp_z2, abs=1e-12)

    assert nonhydro_params.bqdr == pytest.approx(df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32)), abs=1e-12)
    assert nonhydro_params.aqdr == pytest.approxdf32 / dz32 - bqdr * dz32, abs=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")]
)
def test_nonhydro_predictor_step(
    icon_grid,
):
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams()

    diagnostic_state = DiagnosticState(
        hdef_ic=None,
        div_ic=None,
        dwdx=None,
        dwdy=None,
        vt=None,
        vn_ie=None,
        w_concorr_c=None,
        ddt_w_adv_pc_before=None,
        ddt_vn_apc_pc_before=None,
        ntnd=None,
    diagnostic_state_nonhydro = DiagnosticStateNonHydro(
        ddt_vn_phy=sp.ddt_vn_phy(),
        ddt_vn_adv=sp.ddt_vn_adv(),
        ntl1=sp.ntl1(),
        ntl2=sp.ntl2(),
        rho_incr=sp.rho_incr(),
        vn_incr=sp.vn_incr(),
        exner_incr=sp.exner_incr()
    )
    prognostic_state = PrognosticState(
        w=sp.w(),
        vn=None,
        exner_pressure=None,
        theta_v=sp.theta_v(),
        rho=sp.rho(),
        exner=sp.exner()
    )

    interpolation_state = InterpolationState(
        e_bln_c_s=None,
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=None,
        geofac_n2s=None,
        geofac_grg_x=None,
        geofac_grg_y=None,
        nudgecoeff_e=None,
        c_lin_e=savepoint.c_lin_e(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        c_intp=savepoint.c_intp(),
        geofac_rot=savepoint.geofac_rot(),
    )

    metric_state = MetricState(
        mask_hdiff=None,
        theta_ref_mc=sp.theta_ref_mc(),
        wgtfac_c=sp.wgtfac_c(),
        zd_intcoef=None,
        zd_vertidx=None,
        zd_diffcoef=None,
        coeff_gradekin=None,
        ddqz_z_full_e=sp.ddqz_z_full_e(),
        wgtfac_e=sp.wgtfac_e(),
        wgtfacq_e=sp.wgtfacq_e(),
        ddxn_z_full=sp.ddxn_z_full(),
        ddxt_z_full=None, #yes
        ddqz_z_half=sp.ddqz_z_half(),
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=savepoint.exner_exfac,
        exner_ref_mc=savepoint.exner_ref_mc,
        wgtfacq_c=savepoint.wgtfacq_c,
        inv_ddqz_z_full=savepoint.inv_ddqz_z_full,
        rho_ref_mc=savepoint.rho_ref_mc,
        theta_ref_mc=savepoint.theta_ref_mc,
        vwind_expl_wgt=savepoint.vwind_expl_wgt,
        d_exner_dz_ref_ic=savepoint.d_exner_dz_ref_ic,
        ddqz_z_half=savepoint.ddqz_z_half,
        theta_ref_ic=savepoint.theta_ref_ic,
        d2dexdz2_fac1_mc=savepoint.d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=savepoint.d2dexdz2_fac2_mc,
        vwind_impl_wgt=savepoint.vwind_impl_wgt,
        bdy_halo_c=savepoint.bdy_halo_c,
        ipeidx_dsl=savepoint.ipeidx_dsl,
        pg_exdist=savepoint.pg_exdist,
        hmask_dd3d=savepoint.hmask_dd3d,
        scalfac_dd3d=savepoint.scalfac_dd3d,
        rayleigh_w=savepoint.rayleigh_w,
        rho_ref_me=savepoint.rho_ref_me,
        theta_ref_me=savepoint.theta_ref_me,
        zdiff_gradp=savepoint.zdiff_gradp,
        mask_prog_halo_c=savepoint.mask_prog_halo_c,
        mask_hdiff=savepoint.mask_hdiff
    )

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state=metric_state,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params = vertical_params
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )


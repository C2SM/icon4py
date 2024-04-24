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

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import NonHydrostaticConfig
from icon4py.model.atmosphere.dycore.state_utils.states import (
    InterpolationState,
    MetricStateNonHydro,
)
from icon4py.model.common.dimension import CEDim
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field
from icon4py.model.common.test_utils.serialbox_utils import InterpolationSavepoint, MetricSavepoint


def mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps):
    """Create configuration matching the mch_chR04b09_dsl experiment."""
    config = NonHydrostaticConfig(
        ndyn_substeps_var=ndyn_substeps,
        divdamp_order=24,
        iau_wgt_dyn=1.0,
        divdamp_fac=0.004,
        max_nudging_coeff=0.075,
    )
    return config


def exclaim_ape_nonhydrostatic_config(ndyn_substeps):
    """Create configuration for EXCLAIM APE experiment."""
    return NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=24,
        ndyn_substeps_var=ndyn_substeps,
    )


def construct_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_nonhydrostatic_config(ndyn_substeps)


def construct_interpolation_state_for_nonhydro(
    savepoint: InterpolationSavepoint,
) -> InterpolationState:
    grg = savepoint.geofac_grg()
    return InterpolationState(
        c_lin_e=savepoint.c_lin_e(),
        c_intp=savepoint.c_intp(),
        e_flx_avg=savepoint.e_flx_avg(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        geofac_rot=savepoint.geofac_rot(),
        pos_on_tplane_e_1=savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=savepoint.pos_on_tplane_e_y(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        e_bln_c_s=as_1D_sparse_field(savepoint.e_bln_c_s(), CEDim),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=as_1D_sparse_field(savepoint.geofac_div(), CEDim),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
    )


def construct_nh_metric_state(savepoint: MetricSavepoint, num_k_lev) -> MetricStateNonHydro:
    return MetricStateNonHydro(
        bdy_halo_c=savepoint.bdy_halo_c(),
        mask_prog_halo_c=savepoint.mask_prog_halo_c(),
        rayleigh_w=savepoint.rayleigh_w(),
        exner_exfac=savepoint.exner_exfac(),
        exner_ref_mc=savepoint.exner_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        wgtfacq_c=savepoint.wgtfacq_c_dsl(),
        inv_ddqz_z_full=savepoint.inv_ddqz_z_full(),
        rho_ref_mc=savepoint.rho_ref_mc(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        vwind_expl_wgt=savepoint.vwind_expl_wgt(),
        d_exner_dz_ref_ic=savepoint.d_exner_dz_ref_ic(),
        ddqz_z_half=savepoint.ddqz_z_half(),
        theta_ref_ic=savepoint.theta_ref_ic(),
        d2dexdz2_fac1_mc=savepoint.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=savepoint.d2dexdz2_fac2_mc(),
        rho_ref_me=savepoint.rho_ref_me(),
        theta_ref_me=savepoint.theta_ref_me(),
        ddxn_z_full=savepoint.ddxn_z_full(),
        zdiff_gradp=savepoint.zdiff_gradp(),
        vertoffset_gradp=savepoint.vertoffset_gradp(),
        ipeidx_dsl=savepoint.ipeidx_dsl(),
        pg_exdist=savepoint.pg_exdist(),
        ddqz_z_full_e=savepoint.ddqz_z_full_e(),
        ddxt_z_full=savepoint.ddxt_z_full(),
        wgtfac_e=savepoint.wgtfac_e(),
        wgtfacq_e=savepoint.wgtfacq_e_dsl(num_k_lev),
        vwind_impl_wgt=savepoint.vwind_impl_wgt(),
        hmask_dd3d=savepoint.hmask_dd3d(),
        scalfac_dd3d=savepoint.scalfac_dd3d(),
        coeff1_dwdz=savepoint.coeff1_dwdz(),
        coeff2_dwdz=savepoint.coeff2_dwdz(),
        coeff_gradekin=savepoint.coeff_gradekin(),
    )

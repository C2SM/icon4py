# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Assemble the solve-nonhydro (dycore) granule states from the factory sources.

Mirrors the assembly in
`icon4py.model.standalone_driver.driver_utils.initialize_granules` (kept as a local copy
to avoid a `bindings -> standalone_driver` dependency edge). Every field is a direct
`.get(KEY)` from the factories: unlike the v1 dycore wrapper, no host-side transforms
(rbf transpose, wgtfacq flip, vertoffset/index2offset, nflat -1) are needed here because
the factories already produce the icon4py-internal convention.
"""

from icon4py.bindings.v2 import factory_setup
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes


def assemble_dycore_interpolation_state(interpolation) -> dycore_states.InterpolationState:
    return dycore_states.InterpolationState(
        c_lin_e=interpolation.get(interpolation_attributes.C_LIN_E),
        c_intp=interpolation.get(interpolation_attributes.CELL_AW_VERTS),
        e_flx_avg=interpolation.get(interpolation_attributes.E_FLX_AVG),
        geofac_grdiv=interpolation.get(interpolation_attributes.GEOFAC_GRDIV),
        geofac_rot=interpolation.get(interpolation_attributes.GEOFAC_ROT),
        pos_on_tplane_e_1=interpolation.get(interpolation_attributes.POS_ON_TPLANE_E_X),
        pos_on_tplane_e_2=interpolation.get(interpolation_attributes.POS_ON_TPLANE_E_Y),
        rbf_vec_coeff_e=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_E),
        e_bln_c_s=interpolation.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation.get(interpolation_attributes.NUDGECOEFFS_E),
    )


def assemble_dycore_metric_state(metrics) -> dycore_states.MetricStateNonHydro:
    return dycore_states.MetricStateNonHydro(
        mask_prog_halo_c=metrics.get(metrics_attributes.MASK_PROG_HALO_C),
        rayleigh_w=metrics.get(metrics_attributes.RAYLEIGH_W),
        time_extrapolation_parameter_for_exner=metrics.get(metrics_attributes.EXNER_EXFAC),
        reference_exner_at_cells_on_model_levels=metrics.get(metrics_attributes.EXNER_REF_MC),
        wgtfac_c=metrics.get(metrics_attributes.WGTFAC_C),
        wgtfacq_c=metrics.get(metrics_attributes.WGTFACQ_C),
        inv_ddqz_z_full=metrics.get(metrics_attributes.INV_DDQZ_Z_FULL),
        reference_rho_at_cells_on_model_levels=metrics.get(metrics_attributes.RHO_REF_MC),
        reference_theta_at_cells_on_model_levels=metrics.get(metrics_attributes.THETA_REF_MC),
        exner_w_explicit_weight_parameter=metrics.get(
            metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER
        ),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics.get(
            metrics_attributes.D_EXNER_DZ_REF_IC
        ),
        ddqz_z_half=metrics.get(metrics_attributes.DDQZ_Z_HALF),
        reference_theta_at_cells_on_half_levels=metrics.get(metrics_attributes.THETA_REF_IC),
        d2dexdz2_fac1_mc=metrics.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        d2dexdz2_fac2_mc=metrics.get(metrics_attributes.D2DEXDZ2_FAC2_MC),
        reference_rho_at_edges_on_model_levels=metrics.get(metrics_attributes.RHO_REF_ME),
        reference_theta_at_edges_on_model_levels=metrics.get(metrics_attributes.THETA_REF_ME),
        ddxn_z_full=metrics.get(metrics_attributes.DDXN_Z_FULL),
        zdiff_gradp=metrics.get(metrics_attributes.ZDIFF_GRADP),
        vertoffset_gradp=metrics.get(metrics_attributes.VERTOFFSET_GRADP),
        nflat_gradp=metrics.get_int32(metrics_attributes.NFLAT_GRADP),
        pg_exdist=metrics.get(metrics_attributes.PG_EXDIST_DSL),
        ddqz_z_full_e=metrics.get(metrics_attributes.DDQZ_Z_FULL_E),
        ddxt_z_full=metrics.get(metrics_attributes.DDXT_Z_FULL),
        wgtfac_e=metrics.get(metrics_attributes.WGTFAC_E),
        wgtfacq_e=metrics.get(metrics_attributes.WGTFACQ_E),
        exner_w_implicit_weight_parameter=metrics.get(
            metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER
        ),
        horizontal_mask_for_3d_divdamp=metrics.get(
            metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP
        ),
        scaling_factor_for_3d_divdamp=metrics.get(metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP),
        coeff1_dwdz=metrics.get(metrics_attributes.COEFF1_DWDZ),
        coeff2_dwdz=metrics.get(metrics_attributes.COEFF2_DWDZ),
        coeff_gradekin=metrics.get(metrics_attributes.COEFF_GRADEKIN),
    )


def assemble_dycore_states(
    sources: factory_setup.StaticFieldSources,
) -> tuple[dycore_states.InterpolationState, dycore_states.MetricStateNonHydro]:
    return (
        assemble_dycore_interpolation_state(sources.interpolation),
        assemble_dycore_metric_state(sources.metrics),
    )

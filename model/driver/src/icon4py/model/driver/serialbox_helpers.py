# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.atmosphere.diffusion import diffusion_states as diffus_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import serialbox as sb


"""
Construct state objects from serialized data by reading from IconSavepoint s.

This is a preliminary module which should be deleted once we become independent
from the serialized ICON data.
Code is essentially duplicated from the model/atmosphere/xx packages test functionality in order
to get the dependencies right.
"""


def construct_interpolation_state_for_diffusion(
    savepoint: sb.InterpolationSavepoint,
) -> diffus_states.DiffusionInterpolationState:
    grg = savepoint.geofac_grg()
    return diffus_states.DiffusionInterpolationState(
        e_bln_c_s=data_alloc.as_1D_sparse_field(savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=data_alloc.as_1D_sparse_field(savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
    )


def construct_metric_state_for_diffusion(
    savepoint: sb.MetricSavepoint,
) -> diffus_states.DiffusionMetricState:
    return diffus_states.DiffusionMetricState(
        mask_hdiff=savepoint.mask_hdiff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertoffset=savepoint.zd_vertoffset(),
        zd_diffcoef=savepoint.zd_diffcoef(),
    )


def construct_diagnostics_for_diffusion(
    savepoint: sb.IconDiffusionInitSavepoint,
) -> diffus_states.DiffusionDiagnosticState:
    return diffus_states.DiffusionDiagnosticState(
        hdef_ic=savepoint.hdef_ic(),
        div_ic=savepoint.div_ic(),
        dwdx=savepoint.dwdx(),
        dwdy=savepoint.dwdy(),
    )

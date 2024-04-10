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

from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.dimension import CEDim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, zero_field
from icon4py.model.common.test_utils.serialbox_utils import (
    IconGridSavepoint,
    IconDiffusionInitSavepoint,
    InterpolationSavepoint,
    MetricSavepoint,
)


"""
Construct state objects from serialized data by reading from IconSavepoint s.

This is a preliminary module which should be deleted once we become independent
from the serialized ICON data.
Code is essentially duplicated from the model/atmosphere/xx packages test functionality in order
to get the dependencies right.
"""


def construct_interpolation_state_for_diffusion(
    savepoint: InterpolationSavepoint,
) -> DiffusionInterpolationState:
    grg = savepoint.geofac_grg()
    return DiffusionInterpolationState(
        e_bln_c_s=as_1D_sparse_field(savepoint.e_bln_c_s(), CEDim),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=as_1D_sparse_field(savepoint.geofac_div(), CEDim),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
    )


def construct_metric_state_for_diffusion(savepoint: MetricSavepoint) -> DiffusionMetricState:
    return DiffusionMetricState(
        mask_hdiff=savepoint.mask_hdiff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertoffset=savepoint.zd_vertoffset(),
        zd_diffcoef=savepoint.zd_diffcoef(),
    )


def construct_diagnostics_for_diffusion(
    savepoint: IconDiffusionInitSavepoint,
    grid_savepoint: IconGridSavepoint,
) -> DiffusionDiagnosticState:
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    dwdx = savepoint.dwdx() if savepoint.dwdx() else zero_field(grid, CellDim, KDim)
    dwdy = savepoint.dwdy() if savepoint.dwdy() else zero_field(grid, CellDim, KDim)
    return DiffusionDiagnosticState(
        hdef_ic=savepoint.hdef_ic(),
        div_ic=savepoint.div_ic(),
        dwdx=dwdx,
        dwdy=dwdy,
    )

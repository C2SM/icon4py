# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The static states of the tmx component, assembled from the field factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes


if TYPE_CHECKING:
    from icon4py.model.common.states import factory as states_factory


def build_metric_state(metrics_source: states_factory.FieldSource) -> tmx_states.TmxMetricState:
    return tmx_states.TmxMetricState(
        ddqz_z_full=metrics_source.get(metrics_attributes.DDQZ_Z_FULL),
        inv_ddqz_z_full=metrics_source.get(metrics_attributes.INV_DDQZ_Z_FULL),
        ddqz_z_half=metrics_source.get(metrics_attributes.DDQZ_Z_HALF),
        inv_ddqz_z_half=metrics_source.get(metrics_attributes.INV_DDQZ_Z_HALF),
        inv_ddqz_z_full_e=metrics_source.get(metrics_attributes.INV_DDQZ_Z_FULL_E),
        inv_ddqz_z_half_e=metrics_source.get(metrics_attributes.INV_DDQZ_Z_HALF_E),
        inv_ddqz_z_half_v=metrics_source.get(metrics_attributes.INV_DDQZ_Z_HALF_V),
        wgtfac_c=metrics_source.get(metrics_attributes.WGTFAC_C),
        wgtfac_e=metrics_source.get(metrics_attributes.WGTFAC_E),
        wgtfacq_c=metrics_source.get(metrics_attributes.WGTFACQ_C),
        wgtfacq1_c=metrics_source.get(metrics_attributes.WGTFACQ1_C),
        wgtfacq_e=metrics_source.get(metrics_attributes.WGTFACQ_E),
        wgtfacq1_e=metrics_source.get(metrics_attributes.WGTFACQ1_E),
        geopot_agl_ifc=metrics_source.get(metrics_attributes.GEOPOT_AGL_IFC),
        height_above_ground=metrics_source.get(metrics_attributes.HEIGHT_ABOVE_GROUND),
    )


def build_interpolation_state(
    interpolation_source: states_factory.FieldSource,
) -> tmx_states.TmxInterpolationState:
    return tmx_states.TmxInterpolationState(
        c_lin_e=interpolation_source.get(interpolation_attributes.C_LIN_E),
        e_bln_c_s=interpolation_source.get(interpolation_attributes.E_BLN_C_S),
        geofac_div=interpolation_source.get(interpolation_attributes.GEOFAC_DIV),
        cells_aw_verts=interpolation_source.get(interpolation_attributes.CELL_AW_VERTS),
        rbf_coeff_v1=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_v2=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        rbf_coeff_e=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_E),
        rbf_coeff_c1=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_C1),
        rbf_coeff_c2=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_C2),
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Assemble the diffusion granule states and geometry params from the factory sources.

Mirrors the assembly in
`icon4py.model.standalone_driver.driver_utils.initialize_granules` (kept as a local copy
to avoid a `bindings -> standalone_driver` dependency edge).
"""

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.common.grid import geometry_attributes as geometry_meta, states as grid_states
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import factory as states_factory

from icon4py.bindings.v2 import factory_setup


def assemble_cell_params(geometry) -> grid_states.CellParams:
    return grid_states.CellParams(
        cell_center_lat=geometry.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry.get(geometry_meta.CELL_LON),
        area=geometry.get(geometry_meta.CELL_AREA),
        mean_cell_area=geometry.get(
            geometry_meta.MEAN_CELL_AREA, states_factory.RetrievalType.SCALAR
        ),
    )


def assemble_edge_params(geometry) -> grid_states.EdgeParams:
    return grid_states.EdgeParams(
        tangent_orientation=geometry.get(geometry_meta.TANGENT_ORIENTATION),
        inverse_primal_edge_lengths=geometry.get(f"inverse_of_{geometry_meta.EDGE_LENGTH}"),
        inverse_dual_edge_lengths=geometry.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"),
        inverse_vertex_vertex_lengths=geometry.get(
            f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
        ),
        primal_normal_vert_x=geometry.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
        primal_normal_vert_y=geometry.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        dual_normal_vert_x=geometry.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
        dual_normal_vert_y=geometry.get(geometry_meta.EDGE_TANGENT_VERTEX_V),
        primal_normal_cell_x=geometry.get(geometry_meta.EDGE_NORMAL_CELL_U),
        dual_normal_cell_x=geometry.get(geometry_meta.EDGE_TANGENT_CELL_U),
        primal_normal_cell_y=geometry.get(geometry_meta.EDGE_NORMAL_CELL_V),
        dual_normal_cell_y=geometry.get(geometry_meta.EDGE_TANGENT_CELL_V),
        edge_areas=geometry.get(geometry_meta.EDGE_AREA),
        coriolis_frequency=geometry.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_center_lat=geometry.get(geometry_meta.EDGE_LAT),
        edge_center_lon=geometry.get(geometry_meta.EDGE_LON),
        primal_normal_x=geometry.get(geometry_meta.EDGE_NORMAL_U),
        primal_normal_y=geometry.get(geometry_meta.EDGE_NORMAL_V),
    )


def assemble_diffusion_interpolation_state(
    interpolation,
) -> diffusion_states.DiffusionInterpolationState:
    return diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation.get(interpolation_attributes.NUDGECOEFFS_E),
    )


def assemble_diffusion_metric_state(metrics) -> diffusion_states.DiffusionMetricState:
    return diffusion_states.DiffusionMetricState(
        theta_ref_mc=metrics.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics.get(metrics_attributes.ZD_INTCOEF),
        zd_vertoffset=metrics.get(metrics_attributes.ZD_VERTOFFSET),
        zd_diffcoef=metrics.get(metrics_attributes.ZD_DIFFCOEF),
    )


def assemble_diffusion_states(
    sources: factory_setup.StaticFieldSources,
) -> tuple[diffusion_states.DiffusionInterpolationState, diffusion_states.DiffusionMetricState]:
    return (
        assemble_diffusion_interpolation_state(sources.interpolation),
        assemble_diffusion_metric_state(sources.metrics),
    )

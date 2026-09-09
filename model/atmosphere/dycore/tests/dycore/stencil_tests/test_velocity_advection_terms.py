# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.velocity_advection_terms import (
    _compute_extra_diffusion_for_w,
    _compute_interpolated_horizontal_advection_of_w,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import reference_funcs, stencil_tests
from icon4py.model.testing.reference_funcs import interpolate_to_cell_center_numpy

from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_interpolate_cell_field_to_half_levels import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)
from .test_mo_math_divrot_rot_vertex_ri_dsl import mo_math_divrot_rot_vertex_ri_dsl_numpy


def interpolate_vn_to_half_levels_numpy(wgtfac_e: np.ndarray, vn: np.ndarray) -> np.ndarray:
    nlev = vn.shape[1]
    vn_ie = np.zeros((vn.shape[0], nlev + 1))
    w = wgtfac_e[:, 1:nlev]
    vn_ie[:, 1:nlev] = w * vn[:, 1:nlev] + (1.0 - w) * vn[:, 0 : nlev - 1]
    vn_ie[:, 0] = vn[:, 0]
    return vn_ie


def compute_horizontal_kinetic_energy_at_edges_numpy(vn: np.ndarray, vt: np.ndarray) -> np.ndarray:
    return 0.5 * (vn * vn + vt * vt)


def extrapolate_to_surface_numpy(wgtfacq_e: np.ndarray, vn: np.ndarray) -> np.ndarray:
    vn_k_minus_1 = vn[:, -1]
    vn_k_minus_2 = vn[:, -2]
    vn_k_minus_3 = vn[:, -3]
    wgtfacq_e_k_minus_1 = wgtfacq_e[:, -1]
    wgtfacq_e_k_minus_2 = wgtfacq_e[:, -2]
    wgtfacq_e_k_minus_3 = wgtfacq_e[:, -3]
    vn_at_surface = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )
    return vn_at_surface


def compute_diagnostics_from_normal_wind_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    tangential_wind_on_half_levels: np.ndarray,
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
    wgtfac_e: np.ndarray,
    wgtfacq_e: np.ndarray,
    ddxn_z_full: np.ndarray,
    ddxt_z_full: np.ndarray,
    skip_compute_predictor_vertical_advection: bool,
    nlev: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tangential_wind = reference_funcs.compute_tangential_wind_numpy(
        connectivities, vn, rbf_vec_coeff_e
    )
    horizontal_kinetic_energy_at_edges_on_model_levels = (
        compute_horizontal_kinetic_energy_at_edges_numpy(vn, tangential_wind)
    )
    vn_on_half_levels = interpolate_vn_to_half_levels_numpy(wgtfac_e, vn)
    vn_on_half_levels[:, nlev] = extrapolate_to_surface_numpy(wgtfacq_e, vn)

    tangential_wind_on_half_levels = tangential_wind_on_half_levels.copy()
    if not skip_compute_predictor_vertical_advection:
        tangential_wind_on_half_levels[:, :nlev] = interpolate_vt_to_interface_edges_numpy(
            wgtfac_e, tangential_wind
        )[:, :nlev]

    contravariant_correction_at_edges_on_model_levels = compute_contravariant_correction_numpy(
        vn, ddxn_z_full, ddxt_z_full, tangential_wind
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    )


def interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    contravariant_correction_at_edges_on_model_levels: np.ndarray,
    e_bln_c_s: np.ndarray,
    wgtfac_c: np.ndarray,
    nflatlev: int,
    nlev: int,
) -> np.ndarray:
    k = np.arange(nlev)

    contravariant_correction_at_cells_model_levels = interpolate_to_cell_center_numpy(
        connectivities, contravariant_correction_at_edges_on_model_levels, e_bln_c_s
    )

    return np.where(
        k >= nflatlev + 1,
        interpolate_cell_field_to_half_levels_vp_numpy(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        )[:, :-1],
        0.0,
    )


def interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
) -> np.ndarray:
    num_rows, num_cols = contravariant_corrected_w_at_cells_on_half_levels.shape
    contravariant_corrected_w_with_surface = np.zeros((num_rows, num_cols + 1))
    contravariant_corrected_w_with_surface[:, :-1] = (
        contravariant_corrected_w_at_cells_on_half_levels
    )
    return 0.5 * (
        contravariant_corrected_w_with_surface[:, :-1]
        + contravariant_corrected_w_with_surface[:, 1:]
    )


def compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
    *,
    w: np.ndarray,
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
    nlev: int,
    end_index_of_damping_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_rows, num_cols = contravariant_correction_at_cells_on_half_levels.shape

    k = np.arange(num_cols)
    condition = (np.maximum(2, end_index_of_damping_layer - 2) <= k) & (k < nlev - 3)

    contravariant_corrected_w_at_cells_on_half_levels = (
        w - contravariant_correction_at_cells_on_half_levels
    )

    cfl_clipping = np.where(
        (np.abs(contravariant_corrected_w_at_cells_on_half_levels) > cfl_w_limit * ddqz_z_half)
        & condition,
        np.ones([num_rows, num_cols]),
        np.zeros_like(contravariant_corrected_w_at_cells_on_half_levels),
    )
    vertical_cfl = np.where(
        cfl_clipping == 1.0,
        contravariant_corrected_w_at_cells_on_half_levels * dtime / ddqz_z_half,
        0.0,
    )
    contravariant_corrected_w_at_cells_on_half_levels = np.where(
        (cfl_clipping == 1.0) & (vertical_cfl < -0.85),
        -0.85 * ddqz_z_half / dtime,
        contravariant_corrected_w_at_cells_on_half_levels,
    )
    contravariant_corrected_w_at_cells_on_half_levels = np.where(
        (cfl_clipping == 1.0) & (vertical_cfl > 0.85),
        0.85 * ddqz_z_half / dtime,
        contravariant_corrected_w_at_cells_on_half_levels,
    )

    return contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl


def compute_horizontal_advection_of_w_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    w: np.ndarray,
    tangential_wind_on_half_levels: np.ndarray,
    vn_on_half_levels: np.ndarray,
    c_intp: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    tangent_orientation: np.ndarray,
) -> np.ndarray:
    w_at_vertices = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
        connectivities, w, c_intp
    )

    return compute_horizontal_advection_term_for_vertical_velocity_numpy(
        connectivities=connectivities,
        vn_ie=vn_on_half_levels,
        inv_dual_edge_length=inv_dual_edge_length,
        w=w,
        z_vt_ie=tangential_wind_on_half_levels,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        z_w_v=w_at_vertices,
    )


def add_extra_diffusion_for_w_approaching_cfl_wihtout_levmask_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    cfl_clipping: np.ndarray,
    owner_mask: np.ndarray,
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    vertical_wind_advective_tendency: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    owner_mask = np.expand_dims(owner_mask, axis=-1)
    area = np.expand_dims(area, axis=-1)
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

    difcoef = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(contravariant_corrected_w_at_cells_on_half_levels) * dtime / ddqz_z_half
            - cfl_w_limit * dtime,
        ),
        0,
    )

    c2e2cO = connectivities[dims.C2E2CO]
    return np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        vertical_wind_advective_tendency
        + difcoef
        * area
        * np.sum(
            np.where(
                (c2e2cO != -1)[:, :, np.newaxis],
                w[c2e2cO] * geofac_n2s,
                0,
            ),
            axis=1,
        ),
        vertical_wind_advective_tendency,
    )


def compute_advective_vertical_wind_tendency_numpy(
    z_w_con_c: np.ndarray,
    w: np.ndarray,
    coeff1_dwdz: np.ndarray,
    coeff2_dwdz: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    # coeff*_dwdz live on model levels; model level k pairs with half level k
    nlev = coeff1_dwdz.shape[1]
    ddt_w_adv = np.zeros((z_w_con_c.shape[0], nlev + 1))
    c1, c2 = coeff1_dwdz[:, 1:nlev], coeff2_dwdz[:, 1:nlev]
    ddt_w_adv[:, 1:nlev] = -z_w_con_c[:, 1:nlev] * (
        w[:, 0 : nlev - 1] * c1 - w[:, 2 : nlev + 1] * c2 + w[:, 1:nlev] * (c2 - c1)
    )
    return ddt_w_adv


def compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    vertical_wind_advective_tendency: np.ndarray,
    w: np.ndarray,
    horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
    cfl_clipping: np.ndarray,
    coeff1_dwdz: np.ndarray,
    coeff2_dwdz: np.ndarray,
    e_bln_c_s: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    owner_mask: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
    nlev: int,
    end_index_of_damping_layer: int,
) -> np.ndarray:
    k = np.arange(nlev)

    condition1 = k >= 1
    vertical_wind_advective_tendency = np.where(
        condition1,
        compute_advective_vertical_wind_tendency_numpy(
            contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
        )[:, :-1],
        vertical_wind_advective_tendency,
    )

    vertical_wind_advective_tendency = np.where(
        condition1,
        vertical_wind_advective_tendency
        + compute_interpolated_horizontal_advection_of_w_numpy(
            connectivities,
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels[:, :-1],
        ),
        vertical_wind_advective_tendency,
    )

    condition2 = (np.maximum(2, end_index_of_damping_layer - 2) <= k) & (k < nlev - 3)

    return np.where(
        condition2,
        add_extra_diffusion_for_w_approaching_cfl_wihtout_levmask_numpy(
            connectivities=connectivities,
            cfl_clipping=cfl_clipping,
            owner_mask=owner_mask,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            w=w[:, :-1],
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        ),
        vertical_wind_advective_tendency,
    )


def _compute_advective_normal_wind_tendency_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
    coeff_gradekin: np.ndarray,
    horizontal_kinetic_energy_at_cells_on_model_levels: np.ndarray,
    upward_vorticity_at_vertices: np.ndarray,
    tangential_wind: np.ndarray,
    coriolis_frequency: np.ndarray,
    c_lin_e: np.ndarray,
    contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
    vn_on_half_levels: np.ndarray,
    ddqz_z_full_e: np.ndarray,
) -> np.ndarray:
    e2c = connectivities[dims.E2C]
    horizontal_kinetic_energy_at_cells_on_model_levels_e2c = (
        horizontal_kinetic_energy_at_cells_on_model_levels[e2c]
    )
    coeff_gradekin = np.expand_dims(coeff_gradekin, axis=-1)
    coriolis_frequency = np.expand_dims(coriolis_frequency, axis=-1)
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)

    return -(
        (coeff_gradekin[:, 0] - coeff_gradekin[:, 1])
        * horizontal_kinetic_energy_at_edges_on_model_levels
        + (
            -coeff_gradekin[:, 0] * horizontal_kinetic_energy_at_cells_on_model_levels_e2c[:, 0]
            + coeff_gradekin[:, 1] * horizontal_kinetic_energy_at_cells_on_model_levels_e2c[:, 1]
        )
        + tangential_wind
        * (
            coriolis_frequency
            + 0.5 * np.sum(upward_vorticity_at_vertices[connectivities[dims.E2V]], axis=1)
        )
        + np.sum(contravariant_corrected_w_at_cells_on_model_levels[e2c] * c_lin_e, axis=1)
        * (vn_on_half_levels[:, :-1] - vn_on_half_levels[:, 1:])
        / ddqz_z_full_e
    )


def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    c_lin_e: np.ndarray,
    contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
    ddqz_z_full_e: np.ndarray,
    area_edge: np.ndarray,
    tangent_orientation: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    upward_vorticity_at_vertices: np.ndarray,
    geofac_grdiv: np.ndarray,
    vn: np.ndarray,
    normal_wind_advective_tendency: np.ndarray,
    cfl_w_limit: ta.wpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    area_edge = np.expand_dims(area_edge, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    e2c = connectivities[dims.E2C]
    contravariant_corrected_w_at_edges_on_model_levels = np.sum(
        np.where(
            (e2c != -1)[:, :, np.newaxis],
            c_lin_e * contravariant_corrected_w_at_cells_on_model_levels[e2c],
            0,
        ),
        axis=1,
    )

    difcoef = np.where(
        (np.abs(contravariant_corrected_w_at_edges_on_model_levels) > cfl_w_limit * ddqz_z_full_e),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(contravariant_corrected_w_at_edges_on_model_levels) * dtime / ddqz_z_full_e
            - cfl_w_limit * dtime,
        ),
        np.zeros_like(vn),
    )
    e2v = connectivities[dims.E2V]
    e2c2eo = connectivities[dims.E2C2EO]
    return np.where(
        (np.abs(contravariant_corrected_w_at_edges_on_model_levels) > cfl_w_limit * ddqz_z_full_e),
        normal_wind_advective_tendency
        + difcoef
        * area_edge
        * (
            np.sum(
                np.where(
                    (e2c2eo != -1)[:, :, np.newaxis],
                    geofac_grdiv * vn[e2c2eo],
                    0,
                ),
                axis=1,
            )
            + tangent_orientation
            * inv_primal_edge_length
            * (upward_vorticity_at_vertices[e2v][:, 1] - upward_vorticity_at_vertices[e2v][:, 0])
        ),
        normal_wind_advective_tendency,
    )


def compute_advection_in_horizontal_momentum_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    vn: np.ndarray,
    horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
    tangential_wind: np.ndarray,
    coriolis_frequency: np.ndarray,
    contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
    vn_on_half_levels: np.ndarray,
    e_bln_c_s: np.ndarray,
    geofac_rot: np.ndarray,
    coeff_gradekin: np.ndarray,
    c_lin_e: np.ndarray,
    ddqz_z_full_e: np.ndarray,
    area_edge: np.ndarray,
    tangent_orientation: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    geofac_grdiv: np.ndarray,
    cfl_w_limit: ta.wpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
    apply_extra_diffusion_on_vn: bool,
    nlev: int,
    end_index_of_damping_layer: int,
) -> np.ndarray:
    k = np.arange(nlev)

    horizontal_kinetic_energy_at_cells_on_model_levels = interpolate_to_cell_center_numpy(
        connectivities, horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
    )
    upward_vorticity_at_vertices = mo_math_divrot_rot_vertex_ri_dsl_numpy(
        connectivities, vn, geofac_rot
    )

    normal_wind_advective_tendency = _compute_advective_normal_wind_tendency_numpy(
        connectivities=connectivities,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        coeff_gradekin=coeff_gradekin,
        horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
        upward_vorticity_at_vertices=upward_vorticity_at_vertices,
        tangential_wind=tangential_wind,
        coriolis_frequency=coriolis_frequency,
        c_lin_e=c_lin_e,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels=vn_on_half_levels,
        ddqz_z_full_e=ddqz_z_full_e,
    )

    if apply_extra_diffusion_on_vn:
        condition = (np.maximum(2, end_index_of_damping_layer - 2) <= k) & (k < nlev - 4)
        normal_wind_advective_tendency = np.where(
            condition,
            _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask_numpy(
                connectivities=connectivities,
                c_lin_e=c_lin_e,
                contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
                ddqz_z_full_e=ddqz_z_full_e,
                area_edge=area_edge,
                tangent_orientation=tangent_orientation,
                inv_primal_edge_length=inv_primal_edge_length,
                upward_vorticity_at_vertices=upward_vorticity_at_vertices,
                geofac_grdiv=geofac_grdiv,
                vn=vn,
                normal_wind_advective_tendency=normal_wind_advective_tendency,
                cfl_w_limit=cfl_w_limit,
                scalfac_exdiff=scalfac_exdiff,
                dtime=dtime,
            ),
            normal_wind_advective_tendency,
        )

    return normal_wind_advective_tendency


def _restore_outside(
    computed: np.ndarray,
    initial: np.ndarray,
    horizontal: tuple[int, int],
    vertical: tuple[int, int],
) -> np.ndarray:
    """Return `computed` on the given domain and `initial` everywhere else."""
    domain = (slice(*horizontal), slice(*vertical))
    result = initial.copy()
    result[domain] = computed[domain]
    return result


def compute_interpolated_horizontal_advection_of_w_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    e_bln_c_s: np.ndarray,
    horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    c2e = connectivities[dims.C2E]
    return np.sum(
        horizontal_advection_of_w_at_edges_on_half_levels[c2e] * e_bln_c_s,
        axis=1,
    )


def compute_extra_diffusion_for_w_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    area = np.expand_dims(area, axis=-1)
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

    difcoef = scalfac_exdiff * np.minimum(
        0.85 - cfl_w_limit * dtime,
        np.abs(contravariant_corrected_w_at_cells_on_half_levels) * dtime / ddqz_z_half
        - cfl_w_limit * dtime,
    )

    c2e2cO = connectivities[dims.C2E2CO]
    return (
        difcoef
        * area
        * np.sum(
            np.where(
                (c2e2cO != -1)[:, :, np.newaxis],
                w[c2e2cO] * geofac_n2s,
                0,
            ),
            axis=1,
        )
    )


class TestComputeInterpolatedHorizontalAdvectionOfW(stencil_tests.StencilTest):
    PROGRAM = _compute_interpolated_horizontal_advection_of_w
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        e_bln_c_s: np.ndarray,
        horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(
            out=compute_interpolated_horizontal_advection_of_w_numpy(
                connectivities,
                e_bln_c_s,
                horizontal_advection_of_w_at_edges_on_half_levels,
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KHalfDim, dtype=ta.vpfloat
        )
        interpolated_horizontal_advection_of_w = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat
        )

        return dict(
            e_bln_c_s=e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            out=interpolated_horizontal_advection_of_w,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (0, gtx.int32(grid.num_levels + 1)),
            },
        )


@pytest.mark.embedded_remap_error
class TestComputeExtraDiffusionForW(stencil_tests.StencilTest):
    PROGRAM = _compute_extra_diffusion_for_w
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        w: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(
            out=compute_extra_diffusion_for_w_numpy(
                connectivities=connectivities,
                contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
                ddqz_z_half=ddqz_z_half,
                area=area,
                geofac_n2s=geofac_n2s,
                w=w,
                scalfac_exdiff=scalfac_exdiff,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        contravariant_corrected_w_at_cells_on_half_levels = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat
        )
        ddqz_z_half = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, low=0.5, high=1.5, dtype=ta.vpfloat
        )
        area = data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat)
        geofac_n2s = data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=ta.wpfloat)
        w = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        extra_diffusion = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)

        return dict(
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            w=w,
            scalfac_exdiff=ta.wpfloat("10.0"),
            cfl_w_limit=ta.vpfloat("3.0"),
            dtime=ta.wpfloat("2.0"),
            out=extra_diffusion,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (0, gtx.int32(grid.num_levels + 1)),
            },
        )

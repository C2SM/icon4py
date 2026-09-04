# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping, Sequence
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_velocity_advection import (
    compute_velocity_advection_in_corrector_step,
    compute_velocity_advection_in_predictor_step,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import reference_funcs, stencil_tests
from icon4py.model.testing.reference_funcs import interpolate_to_cell_center_numpy

from .test_add_interpolated_horizontal_advection_of_w import (
    add_interpolated_horizontal_advection_of_w_numpy,
)
from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_interpolate_cell_field_to_half_levels import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from .test_interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_vn_ie_numpy,
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_z_kin_hor_e_numpy,
)
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)
from .test_mo_math_divrot_rot_vertex_ri_dsl import mo_math_divrot_rot_vertex_ri_dsl_numpy


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
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_z_kin_hor_e_numpy(
            vn, tangential_wind
        )
    )
    vn_on_half_levels = (
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_vn_ie_numpy(wgtfac_e, vn)
    )
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
        add_interpolated_horizontal_advection_of_w_numpy(
            connectivities,
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels[:, :-1],
            vertical_wind_advective_tendency,
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


@pytest.mark.embedded_remap_error
@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestComputeVelocityAdvectionInPredictorStep(stencil_tests.StencilTest):
    PROGRAM = compute_velocity_advection_in_predictor_step
    OUTPUTS = (
        "tangential_wind",
        "tangential_wind_on_half_levels",
        "vn_on_half_levels",
        "horizontal_kinetic_energy_at_edges_on_model_levels",
        "contravariant_correction_at_edges_on_model_levels",
        "contravariant_correction_at_cells_on_half_levels",
        "vertical_wind_advective_tendency",
        "vertical_cfl",
        "normal_wind_advective_tendency",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "end_index_of_damping_layer",
            "skip_compute_predictor_vertical_advection",
            "apply_extra_diffusion_on_vn",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "start_edge_lateral_boundary_level_5",
            "end_edge_halo_level_2",
            "start_cell_lateral_boundary_level_4",
            "end_cell_halo",
            "start_edge_nudging_level_2",
            "end_edge_local",
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "end_index_of_damping_layer",
            "skip_compute_predictor_vertical_advection",
            "apply_extra_diffusion_on_vn",
        ),
    }

    @pytest.fixture(autouse=True)
    def _xfail_undecidable_subset_split(
        self,
        request: pytest.FixtureRequest,
        static_variant: Sequence[str],
        input_data: dict[str, gtx.Field | state_utils.ScalarType],
    ) -> None:
        # GT4Py's dace `SplitMemlet.can_be_applied` reaches
        # `assert read_right or read_left` in
        # `runners/dace/transformations/splitting_tools.py:515`, where both flags come from
        # `(a < b) == True` on SymPy relationals. That is undecidable, hence False both
        # ways, when the horizontal bounds are symbolic while the vertical ones are static.
        # `setup_program` binds horizontal and vertical bounds together, so the model
        # never compiles this combination.
        vertical_is_static = "vertical_start" in static_variant
        horizontal_is_static = "start_edge_lateral_boundary_level_5" in static_variant
        if (
            "dace" in str(request.config.getoption("backend", ""))
            and vertical_is_static
            and not horizontal_is_static
            and input_data["skip_compute_predictor_vertical_advection"]
            and not input_data["apply_extra_diffusion_on_vn"]
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        "gt4py dace: undecidable SymPy comparison asserted in"
                        " decompose_subset (dace/transformations/splitting_tools.py:515),"
                        " reached from SplitMemlet.can_be_applied with symbolic horizontal"
                        " and static vertical bounds"
                    ),
                    strict=True,
                )
            )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        tangential_wind: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vertical_wind_advective_tendency: np.ndarray,
        vertical_cfl: np.ndarray,
        normal_wind_advective_tendency: np.ndarray,
        vn: np.ndarray,
        w: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        owner_mask: np.ndarray,
        coriolis_frequency: np.ndarray,
        geofac_rot: np.ndarray,
        coeff_gradekin: np.ndarray,
        c_lin_e: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        area_edge: np.ndarray,
        geofac_grdiv: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        skip_compute_predictor_vertical_advection: bool,
        apply_extra_diffusion_on_vn: bool,
        nflatlev: int,
        end_index_of_damping_layer: int,
        start_edge_lateral_boundary_level_5: int,
        end_edge_halo_level_2: int,
        start_cell_lateral_boundary_level_4: int,
        end_cell_halo: int,
        start_edge_nudging_level_2: int,
        end_edge_local: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        nlev = vertical_end

        (
            tangential_wind_new,
            tangential_wind_on_half_levels_new,
            vn_on_half_levels_new,
            horizontal_kinetic_energy_new,
            contravariant_correction_at_edges_new,
        ) = compute_diagnostics_from_normal_wind_numpy(
            connectivities=connectivities,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nlev=nlev,
        )

        contravariant_correction_at_cells_on_half_levels_new = interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
            connectivities=connectivities,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_new,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            nflatlev=nflatlev,
            nlev=nlev,
        )

        (
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            vertical_cfl_new,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            w=w[:, :-1],
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_new,
            ddqz_z_half=ddqz_z_half[:, :-1],
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        vertical_wind_advective_tendency_new = vertical_wind_advective_tendency[:, :-1]
        if not skip_compute_predictor_vertical_advection:
            horizontal_advection_of_w_at_edges_on_half_levels = (
                compute_horizontal_advection_of_w_numpy(
                    connectivities=connectivities,
                    w=w,
                    tangential_wind_on_half_levels=tangential_wind_on_half_levels_new,
                    vn_on_half_levels=vn_on_half_levels_new,
                    c_intp=c_intp,
                    inv_dual_edge_length=inv_dual_edge_length,
                    inv_primal_edge_length=inv_primal_edge_length,
                    tangent_orientation=tangent_orientation,
                )
            )
            vertical_wind_advective_tendency_new = compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
                connectivities=connectivities,
                vertical_wind_advective_tendency=vertical_wind_advective_tendency_new,
                w=w,
                horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
                contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
                cfl_clipping=cfl_clipping,
                coeff1_dwdz=coeff1_dwdz,
                coeff2_dwdz=coeff2_dwdz,
                e_bln_c_s=e_bln_c_s,
                ddqz_z_half=ddqz_z_half[:, :-1],
                area=area,
                geofac_n2s=geofac_n2s,
                owner_mask=owner_mask,
                scalfac_exdiff=scalfac_exdiff,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
                nlev=nlev,
                end_index_of_damping_layer=end_index_of_damping_layer,
            )

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        normal_wind_advective_tendency_new = compute_advection_in_horizontal_momentum_numpy(
            connectivities=connectivities,
            vn=vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_new,
            tangential_wind=tangential_wind_new,
            coriolis_frequency=coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=vn_on_half_levels_new,
            e_bln_c_s=e_bln_c_s,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        edge_slice = (start_edge_lateral_boundary_level_5, end_edge_halo_level_2)
        cell_slice = (start_cell_lateral_boundary_level_4, end_cell_halo)

        return dict(
            tangential_wind=_restore_outside(
                tangential_wind_new, tangential_wind, edge_slice, (vertical_start, vertical_end)
            ),
            tangential_wind_on_half_levels=_restore_outside(
                tangential_wind_on_half_levels_new,
                tangential_wind_on_half_levels,
                edge_slice,
                (vertical_start, vertical_end),
            ),
            vn_on_half_levels=_restore_outside(
                vn_on_half_levels_new,
                vn_on_half_levels,
                edge_slice,
                (vertical_start, vertical_end + 1),
            ),
            horizontal_kinetic_energy_at_edges_on_model_levels=_restore_outside(
                horizontal_kinetic_energy_new,
                horizontal_kinetic_energy_at_edges_on_model_levels,
                edge_slice,
                (vertical_start, vertical_end),
            ),
            contravariant_correction_at_edges_on_model_levels=_restore_outside(
                contravariant_correction_at_edges_new,
                contravariant_correction_at_edges_on_model_levels,
                edge_slice,
                (nflatlev, vertical_end),
            ),
            contravariant_correction_at_cells_on_half_levels=_restore_outside(
                contravariant_correction_at_cells_on_half_levels_new,
                contravariant_correction_at_cells_on_half_levels,
                cell_slice,
                (vertical_start, vertical_end),
            ),
            vertical_wind_advective_tendency=_restore_outside(
                vertical_wind_advective_tendency_new,
                vertical_wind_advective_tendency,
                cell_slice,
                # ICON computes the tendency over jk = 2..nlev
                # (mo_velocity_advection.f90:598), so the top half level is untouched.
                (vertical_start + 1, vertical_end),
            ),
            vertical_cfl=_restore_outside(
                vertical_cfl_new, vertical_cfl, cell_slice, (vertical_start, vertical_end)
            ),
            normal_wind_advective_tendency=_restore_outside(
                normal_wind_advective_tendency_new,
                normal_wind_advective_tendency,
                (start_edge_nudging_level_2, end_edge_local),
                (vertical_start, vertical_end),
            ),
        )

    @stencil_tests.input_data_fixture(
        params=[
            {
                "skip_compute_predictor_vertical_advection": skip,
                "apply_extra_diffusion_on_vn": diffu,
            }
            for skip, diffu in ((False, True), (True, False))
        ],
        ids=lambda param: (
            f"skip_compute_predictor_vertical_advection[{param['skip_compute_predictor_vertical_advection']}]"
            f"-apply_extra_diffusion_on_vn[{param['apply_extra_diffusion_on_vn']}]"
        ),
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        tangential_wind = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim)
        vn_on_half_levels = data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim)
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            dims.CellDim, dims.KHalfDim
        )
        vertical_wind_advective_tendency = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        vertical_cfl = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        normal_wind_advective_tendency = data_alloc.zero_field(dims.EdgeDim, dims.KDim)

        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        w = data_alloc.random_field(dims.CellDim, dims.KHalfDim)

        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim)
        wgtfac_e = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim)
        wgtfacq_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        coeff1_dwdz = data_alloc.random_field(dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(dims.CellDim, dims.KDim)
        c_intp = data_alloc.random_field(dims.VertexDim, dims.V2CDim)
        inv_dual_edge_length = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        inv_primal_edge_length = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        tangent_orientation = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        ddqz_z_half = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        area = data_alloc.random_field(dims.CellDim)
        geofac_n2s = data_alloc.random_field(dims.CellDim, dims.C2E2CODim)
        owner_mask = data_alloc.random_mask(dims.CellDim)
        coriolis_frequency = data_alloc.random_field(dims.EdgeDim)
        geofac_rot = data_alloc.random_field(dims.VertexDim, dims.V2EDim)
        coeff_gradekin = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        c_lin_e = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        # low=0.0 makes sure the simplified stencil produces the same result as the numpy version
        ddqz_z_full_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, low=0.0)
        area_edge = data_alloc.random_field(dims.EdgeDim)
        geofac_grdiv = data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim)

        scalfac_exdiff = 10.0
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime

        # values are set to reflect the MCH ch1 experiment. Changing them changes the runtime
        nflatlev = 5
        end_index_of_damping_layer = 12

        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)

        return dict(
            tangential_wind=tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            vertical_cfl=vertical_cfl,
            normal_wind_advective_tendency=normal_wind_advective_tendency,
            vn=vn,
            w=w,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            coriolis_frequency=coriolis_frequency,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            geofac_grdiv=geofac_grdiv,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=request.param[
                "skip_compute_predictor_vertical_advection"
            ],
            apply_extra_diffusion_on_vn=request.param["apply_extra_diffusion_on_vn"],
            nflatlev=nflatlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
            start_edge_lateral_boundary_level_5=grid.start_index(
                edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
            ),
            end_edge_halo_level_2=grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2)),
            start_cell_lateral_boundary_level_4=grid.start_index(
                cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
            ),
            end_cell_halo=grid.end_index(cell_domain(h_grid.Zone.HALO)),
            start_edge_nudging_level_2=grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2)),
            end_edge_local=grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )


@pytest.mark.embedded_remap_error
@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestComputeVelocityAdvectionInCorrectorStep(stencil_tests.StencilTest):
    PROGRAM = compute_velocity_advection_in_corrector_step
    OUTPUTS = (
        "vertical_wind_advective_tendency",
        "vertical_cfl",
        "normal_wind_advective_tendency",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "end_index_of_damping_layer",
            "apply_extra_diffusion_on_vn",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "start_cell_lateral_boundary_level_4",
            "end_cell_halo",
            "start_edge_nudging_level_2",
            "end_edge_local",
            "vertical_start",
            "vertical_end",
            "end_index_of_damping_layer",
            "apply_extra_diffusion_on_vn",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vertical_wind_advective_tendency: np.ndarray,
        vertical_cfl: np.ndarray,
        normal_wind_advective_tendency: np.ndarray,
        vn: np.ndarray,
        w: np.ndarray,
        tangential_wind: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        e_bln_c_s: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        owner_mask: np.ndarray,
        coriolis_frequency: np.ndarray,
        geofac_rot: np.ndarray,
        coeff_gradekin: np.ndarray,
        c_lin_e: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        area_edge: np.ndarray,
        geofac_grdiv: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        apply_extra_diffusion_on_vn: bool,
        end_index_of_damping_layer: int,
        start_cell_lateral_boundary_level_4: int,
        end_cell_halo: int,
        start_edge_nudging_level_2: int,
        end_edge_local: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        nlev = vertical_end

        horizontal_advection_of_w_at_edges_on_half_levels = compute_horizontal_advection_of_w_numpy(
            connectivities=connectivities,
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
        )

        (
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            vertical_cfl_new,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            w=w[:, :-1],
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels[
                :, :-1
            ],
            ddqz_z_half=ddqz_z_half[:, :-1],
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        vertical_wind_advective_tendency_new = compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
            connectivities=connectivities,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency[:, :-1],
            w=w,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping=cfl_clipping,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half[:, :-1],
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        normal_wind_advective_tendency_new = compute_advection_in_horizontal_momentum_numpy(
            connectivities=connectivities,
            vn=vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            tangential_wind=tangential_wind,
            coriolis_frequency=coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=vn_on_half_levels,
            e_bln_c_s=e_bln_c_s,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        cell_slice = (start_cell_lateral_boundary_level_4, end_cell_halo)
        return dict(
            vertical_wind_advective_tendency=_restore_outside(
                vertical_wind_advective_tendency_new,
                vertical_wind_advective_tendency,
                cell_slice,
                # ICON computes the tendency over jk = 2..nlev
                # (mo_velocity_advection.f90:598), so the top half level is untouched.
                (vertical_start + 1, vertical_end),
            ),
            vertical_cfl=_restore_outside(
                vertical_cfl_new, vertical_cfl, cell_slice, (vertical_start, vertical_end)
            ),
            normal_wind_advective_tendency=_restore_outside(
                normal_wind_advective_tendency_new,
                normal_wind_advective_tendency,
                (start_edge_nudging_level_2, end_edge_local),
                (vertical_start, vertical_end),
            ),
        )

    @stencil_tests.input_data_fixture(
        params=[{"apply_extra_diffusion_on_vn": value} for value in [True, False]],
        ids=lambda param: f"apply_extra_diffusion_on_vn[{param['apply_extra_diffusion_on_vn']}]",
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vertical_wind_advective_tendency = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        vertical_cfl = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        normal_wind_advective_tendency = data_alloc.zero_field(dims.EdgeDim, dims.KDim)

        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        w = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        tangential_wind = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim)
        vn_on_half_levels = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim)
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim
        )

        coeff1_dwdz = data_alloc.random_field(dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(dims.CellDim, dims.KDim)
        c_intp = data_alloc.random_field(dims.VertexDim, dims.V2CDim)
        inv_dual_edge_length = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        inv_primal_edge_length = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        tangent_orientation = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim)
        ddqz_z_half = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        area = data_alloc.random_field(dims.CellDim)
        geofac_n2s = data_alloc.random_field(dims.CellDim, dims.C2E2CODim)
        owner_mask = data_alloc.random_mask(dims.CellDim)
        coriolis_frequency = data_alloc.random_field(dims.EdgeDim)
        geofac_rot = data_alloc.random_field(dims.VertexDim, dims.V2EDim)
        coeff_gradekin = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        c_lin_e = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        # low=0.0 makes sure the simplified stencil produces the same result as the numpy version
        ddqz_z_full_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, low=0.0)
        area_edge = data_alloc.random_field(dims.EdgeDim)
        geofac_grdiv = data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim)

        scalfac_exdiff = 10.0
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime

        # value is set to reflect the MCH ch1 experiment. Changing it changes the runtime
        end_index_of_damping_layer = 12

        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)

        return dict(
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            vertical_cfl=vertical_cfl,
            normal_wind_advective_tendency=normal_wind_advective_tendency,
            vn=vn,
            w=w,
            tangential_wind=tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            coriolis_frequency=coriolis_frequency,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            geofac_grdiv=geofac_grdiv,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            apply_extra_diffusion_on_vn=request.param["apply_extra_diffusion_on_vn"],
            end_index_of_damping_layer=end_index_of_damping_layer,
            start_cell_lateral_boundary_level_4=grid.start_index(
                cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
            ),
            end_cell_halo=grid.end_index(cell_domain(h_grid.Zone.HALO)),
            start_edge_nudging_level_2=grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2)),
            end_edge_local=grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

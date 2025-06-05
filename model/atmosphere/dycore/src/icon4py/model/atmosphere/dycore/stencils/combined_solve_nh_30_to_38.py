# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.accumulate_prep_adv_fields import _accumulate_prep_adv_fields
from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn import _compute_avg_vn
from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn_and_graddiv_vn_and_vt import \
    _compute_avg_vn_and_graddiv_vn_and_vt
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.atmosphere.dycore.stencils.compute_mass_flux import _compute_mass_flux
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.init_two_edge_kdim_fields_with_zero_wp import \
    _init_two_edge_kdim_fields_with_zero_wp
from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_vp import (
    _interpolate_edge_field_to_half_levels_vp,
)

@gtx.field_operator
def _compute_vt_vn_on_half_levels_and_kinetic_energy(
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    skip_compute_predictor_vertical_advection: bool,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    vn_on_half_levels, horizontal_kinetic_energy_at_edges_on_model_levels = concat_where(
        dims.KDim >= 1,
        _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges(
            wgtfac_e, vn, tangential_wind
        ),
        (vn_on_half_levels, horizontal_kinetic_energy_at_edges_on_model_levels),
    )

    tangential_wind_on_half_levels = (
        concat_where(
            dims.KDim >= 1,
            _interpolate_edge_field_to_half_levels_vp(wgtfac_e, tangential_wind),
            tangential_wind_on_half_levels,
        )
        if not skip_compute_predictor_vertical_advection
        else tangential_wind_on_half_levels
    )

    (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    ) = concat_where(
        dims.KDim == 0,
        _compute_horizontal_kinetic_energy(vn, tangential_wind),
        (
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ),
    )

    return (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    )

@gtx.field_operator
def _combined_solve_nh_30_to_38_predictor(
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    nflatlev: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    z_vn_avg, horizontal_gradient_of_normal_wind_divergence, tangential_wind = _compute_avg_vn_and_graddiv_vn_and_vt(
        e_flx_avg,
        vn,
        geofac_grdiv,
        rbf_vec_coeff_e,
    )

    mass_flux_at_edges_on_model_levels, theta_v_flux_at_edges_on_model_levels = _compute_mass_flux(
        rho_at_edges_on_model_levels,
        z_vn_avg,
        ddqz_z_full_e,
        theta_v_at_edges_on_model_levels,
    )

    contravariant_correction_at_edges_on_model_levels = concat_where(
        nflatlev <= dims.KDim,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edges_on_model_levels,
    )

    (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    ) = _compute_vt_vn_on_half_levels_and_kinetic_energy(
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        vn,
        tangential_wind,
        wgtfac_e,
        skip_compute_predictor_vertical_advection,
    )

    return (
        z_vn_avg,
        horizontal_gradient_of_normal_wind_divergence,
        tangential_wind,
        mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels,
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def combined_solve_nh_30_to_38_predictor(
    z_vn_avg: fa.EdgeKField[ta.wpfloat],
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    nflatlev: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _combined_solve_nh_30_to_38_predictor(
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        vn,
        e_flx_avg,
        geofac_grdiv,
        rbf_vec_coeff_e,
        rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels,
        ddqz_z_full_e,
        ddxn_z_full,
        ddxt_z_full,
        wgtfac_e,
        nflatlev,
        skip_compute_predictor_vertical_advection,
        out=(
            z_vn_avg,
            horizontal_gradient_of_normal_wind_divergence,
            tangential_wind,
            mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels,
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )

@gtx.field_operator
def _init_to_zero_and_accumulate_prep_adv_fields(
    z_vn_avg: fa.EdgeKField[ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    r_nsubsteps: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    vn_traj, mass_flx_me = _init_two_edge_kdim_fields_with_zero_wp()
    vn_traj, mass_flx_me = _accumulate_prep_adv_fields(
        z_vn_avg,
        mass_flux_at_edges_on_model_levels,
        vn_traj,
        mass_flx_me,
        r_nsubsteps,
    )

    return vn_traj, mass_flx_me

@gtx.field_operator
def _combined_solve_nh_30_to_38_corrector(
    vn_traj: fa.EdgeKField[ta.wpfloat],
    mass_flx_me: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    at_first_substep: bool,
    r_nsubsteps: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:

    z_vn_avg = _compute_avg_vn(e_flx_avg, vn)

    mass_flux_at_edges_on_model_levels, theta_v_flux_at_edges_on_model_levels = _compute_mass_flux(
        rho_at_edges_on_model_levels,
        z_vn_avg,
        ddqz_z_full_e,
        theta_v_at_edges_on_model_levels,
    )

    vn_traj, mass_flx_me = (
        _init_to_zero_and_accumulate_prep_adv_fields(
            z_vn_avg,
            mass_flux_at_edges_on_model_levels,
            r_nsubsteps,
        )
        if at_first_substep
        else _accumulate_prep_adv_fields(
            z_vn_avg,
            mass_flux_at_edges_on_model_levels,
            vn_traj,
            mass_flx_me,
            r_nsubsteps,
        )
    )

    return z_vn_avg, mass_flux_at_edges_on_model_levels, theta_v_flux_at_edges_on_model_levels, vn_traj, mass_flx_me

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def combined_solve_nh_30_to_38_corrector(
    z_vn_avg: fa.EdgeKField[ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    vn_traj: fa.EdgeKField[ta.wpfloat],
    mass_flx_me: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    at_first_substep: bool,
    r_nsubsteps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _combined_solve_nh_30_to_38_corrector(
        vn_traj,
        mass_flx_me,
        e_flx_avg,
        vn,
        rho_at_edges_on_model_levels,
        ddqz_z_full_e,
        theta_v_at_edges_on_model_levels,
        at_first_substep,
        r_nsubsteps,
        out=(
            z_vn_avg,
            mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels,
            vn_traj,
            mass_flx_me
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

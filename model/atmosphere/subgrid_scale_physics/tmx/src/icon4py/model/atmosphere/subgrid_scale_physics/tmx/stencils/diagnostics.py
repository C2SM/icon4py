# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stencils of the tmx diagnostics.

Ports ``Compute_diagnostics`` (the Smagorinsky diagnostics of the atmosphere,
mo_vdf_atmo.f90 l. 343-482, run by :meth:`Tmx.run_diagnostics`) and
``Update_diagnostics`` (the end-of-step diagnostics, mo_vdf_atmo.f90 l. 487 and
mo_vdf.f90 l. 354, run by :meth:`Tmx.run_update_diagnostics`).

The field operators are grouped into one program per halo-exchange interval and
horizontal dimension, so that the intermediate fields the Fortran writes to
memory stay inside the fused kernel. Field operators are only fused across a
neighbor gather when the gathered field is also an output of the same program
(otherwise the gather would be recomputed once per neighbor).

Bottom rows are selected with ``concat_where(dims.KDim < nlev - 1, ...)`` rather
than an equality test because ``concat_where(dims.KDim == nlev - 1, ...)`` is
broken in GT4Py (GridTools/gt4py#2205), and constant branch values are anchored
to a K-bounded field (``shifted * 0.0 + value``) instead of a bare
``broadcast``: a broadcast has an unbounded K range, which raises "Cannot
compute length of open 'UnitRange'" on the embedded backend and silently
computes the wrong values with gtfn.
"""

import gt4py.next as gtx
from gt4py.next import abs, maximum, minimum, power, sqrt, where  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import E2C, E2C2V, E2C2VDim, KDim
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_wp,
)
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_with_boundaries_wp import (
    _interpolate_cell_field_to_half_levels_with_boundaries_wp,
)
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_cell_half_levels_wp import (
    _interpolate_edge_field_to_cell_half_levels_wp,
)
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_with_boundaries_wp import (
    _interpolate_edge_field_to_half_levels_with_boundaries_wp,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center_wp import (
    _interpolate_to_cell_center_wp,
)
from icon4py.model.common.math.vertical_operations import (
    _compute_vertical_integral,
    average_level_plus1_on_cells,
)
from icon4py.model.common.physics.stencils.compute_brunt_vaisala_frequency import (
    _compute_brunt_vaisala_frequency,
)
from icon4py.model.common.physics.stencils.compute_dry_static_energy import (
    _compute_dry_static_energy,
)
from icon4py.model.common.physics.stencils.compute_virtual_potential_temperature import (
    _compute_virtual_potential_temperature,
)
from icon4py.model.common.physics.thermodynamics import _internal_energy
from icon4py.model.common.type_alias import wpfloat


# ---------------------------------------------------------------------------
# Smagorinsky_init (mo_tmx_smagorinsky.f90): run once, at granule construction
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_smagorinsky_mixing_length(
    dz_ic: fa.CellKField[wpfloat],
    geopot_agl_ic: fa.CellKField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    smag_constant: wpfloat,
    max_turb_scale: wpfloat,
    grav: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the square of the subgrid-scale mixing length for the Smagorinsky model.

    Port of ``compute_mixing_length`` in ICON's ``mo_tmx_smagorinsky.f90``:

        lambda^2 = (Cs * Delta)^2 * (kappa * x_3)^2 / ((Cs * Delta)^2 + (kappa * x_3)^2)
                 = (Cs * Delta * x_3)^2 / ((Cs * Delta / kappa)^2 + x_3^2)

    with Cs the Smagorinsky constant, Delta the filter/grid width (capped at
    ``max_turb_scale``), x_3 the height above ground, and kappa = 0.4 the
    von Karman constant. Reference: Dipankar et al. (2015).

    Args:
        dz_ic: layer thickness centered at half levels (nlev + 1 levels)
        geopot_agl_ic: geopotential above ground at half levels (nlev + 1 levels)
        cell_area: cell area
        smag_constant: Smagorinsky constant Cs
        max_turb_scale: maximum turbulence length scale
        grav: gravitational acceleration

    Returns:
        square of the Smagorinsky mixing length at half levels
    """
    kappa = PhysicsConstants.von_karman

    z_agl = geopot_agl_ic * (wpfloat("1.0") / grav)
    les_filter = smag_constant * minimum(
        max_turb_scale, power(dz_ic * cell_area, wpfloat("0.33333"))
    )
    return (
        (les_filter * z_agl)
        * (les_filter * z_agl)
        / ((les_filter / kappa) * (les_filter / kappa) + z_agl * z_agl)
    )


@gtx.field_operator
def _compute_scaling_factor_louis(
    cell_area: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    Compute the scaling factor for the Louis constant b.

    Port of ``compute_scaling_factor_louis`` in ICON's ``mo_tmx_smagorinsky.f90``.
    The scaling factor is designed to be 1 with an R2B8 setup.

    Args:
        cell_area: cell area

    Returns:
        scaling factor for the Louis constant b
    """
    # Global mean cell area of the R2B8 grid [m^2] (``mean_area_R2B8`` in ICON's
    # mo_tmx_smagorinsky.f90). Defined here because module-level closure constants
    # are not supported by the gtfn backend.
    mean_cell_area_r2b8 = wpfloat("97294071.23714285")
    return mean_cell_area_r2b8 / cell_area


# The two Smagorinsky_init fields are deliberately not fused into one program:
# the Fortran only computes the Louis scaling factor when the Louis stability
# correction is enabled, and both programs run once, at granule construction.
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_smagorinsky_mixing_length(
    dz_ic: fa.CellKField[wpfloat],
    geopot_agl_ic: fa.CellKField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    mixing_length_sq: fa.CellKField[wpfloat],
    smag_constant: wpfloat,
    max_turb_scale: wpfloat,
    grav: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_smagorinsky_mixing_length(
        dz_ic=dz_ic,
        geopot_agl_ic=geopot_agl_ic,
        cell_area=cell_area,
        smag_constant=smag_constant,
        max_turb_scale=max_turb_scale,
        grav=grav,
        out=mixing_length_sq,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_scaling_factor_louis(
    cell_area: fa.CellField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_scaling_factor_louis(
        cell_area=cell_area,
        out=scaling_factor_louis,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics, before the u/v cell exchange
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_thermodynamic_diagnostics(
    temperature: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    grav: wpfloat,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Thermodynamic cell diagnostics of ``Compute_diagnostics`` (mo_vdf_atmo.f90).

    Fuses ``compute_static_energy``, ``get_virtual_potential_temperature``,
    ``vert_intp_full2half_cell_3d`` (rho -> rho_ic) and ``brunt_vaisala_freq``,
    the four cell loops the Fortran runs before the first halo exchange. The
    Brunt-Vaisala frequency reads the virtual potential temperature this
    operator computes.

    Returns:
        dry static energy, virtual potential temperature, air density at half
        levels and squared Brunt-Vaisala frequency
    """
    dry_static_energy = _compute_dry_static_energy(
        temperature=temperature, height_above_ground=height_above_ground, grav=grav
    )
    theta_v = _compute_virtual_potential_temperature(
        virtual_temperature=virtual_temperature, pressure=pressure
    )
    rho_ic = _interpolate_cell_field_to_half_levels_with_boundaries_wp(
        interpolant=rho,
        wgtfac_c=wgtfac_c,
        wgtfacq1_c=wgtfacq1_c,
        wgtfacq_c=wgtfacq_c,
        nlev=nlev,
    )
    bruvais = _compute_brunt_vaisala_frequency(
        theta_v=theta_v, wgtfac_c=wgtfac_c, inv_ddqz_z_half=inv_ddqz_z_half, grav=grav
    )
    return dry_static_energy, theta_v, rho_ic, bruvais


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_thermodynamic_diagnostics(
    temperature: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    dry_static_energy: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    grav: wpfloat,
    nlev: gtx.int32,
    cell_start_nudging: gtx.int32,
    cell_start_lateral_boundary_level_2: gtx.int32,
    cell_start_lateral_boundary_level_3: gtx.int32,
    cell_end_local: gtx.int32,
    cell_end_halo_level_2: gtx.int32,
    vertical_start: gtx.int32,
    vertical_start_interior: gtx.int32,
    vertical_end: gtx.int32,
    vertical_end_half: gtx.int32,
) -> None:
    _compute_thermodynamic_diagnostics(
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        pressure=pressure,
        rho=rho,
        height_above_ground=height_above_ground,
        wgtfac_c=wgtfac_c,
        inv_ddqz_z_half=inv_ddqz_z_half,
        wgtfacq1_c=wgtfacq1_c,
        wgtfacq_c=wgtfacq_c,
        grav=grav,
        nlev=nlev,
        out=(dry_static_energy, theta_v, rho_ic, bruvais),
        domain=(
            # cptgz: the tmx t_domain cells, all full levels
            {
                dims.CellDim: (cell_start_nudging, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            # theta_v: cells rl 3..min_rlcell_int, all full levels
            {
                dims.CellDim: (cell_start_lateral_boundary_level_3, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            # rho_ic: cells rl 2..min_rlcell_int-2, all half levels (the top and
            # bottom rows are extrapolated)
            {
                dims.CellDim: (cell_start_lateral_boundary_level_2, cell_end_halo_level_2),
                dims.KDim: (vertical_start, vertical_end_half),
            },
            # bruvais: cells rl 3..min_rlcell_int, half levels 1..nlev-1 (the top
            # and bottom rows are not computed)
            {
                dims.CellDim: (cell_start_lateral_boundary_level_3, cell_end_local),
                dims.KDim: (vertical_start_interior, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics, after the vertex exchange: edge diagnostics
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_shear_and_div_of_stress(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    w: fa.CellKField[wpfloat],
    vn_ie: fa.EdgeKField[wpfloat],
    vt_ie: fa.EdgeKField[wpfloat],
    w_ie: fa.EdgeKField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Compute shear and divergence of stress at edges of full levels.

    Fuses the ICON TMX subroutines 'compute_velocity_gradient_tensor' and
    'compute_shear' (mo_vdf_atmo.f90). The 3x3 velocity gradient tensor
    (first index: velocity component, second index: derivative direction;
    1: normal, 2: tangential, 3: vertical) is kept in local temporaries and
    contracted into

        shear      = 2 * |S|^2 = 4 * (T_11^2 + T_22^2 + T_33^2)
                     + 2 * (D_12^2 + D_13^2 + D_23^2),  D_ij = T_ij + T_ji
        div_stress = trace(S_ij) = T_11 + T_22 + T_33

    Half-level (interface) input fields (w_vert, w, vn_ie, vt_ie, w_ie) must
    provide num_levels + 1 vertical levels; outputs live on full levels.
    """
    # Normal/tangential velocity components at the four E2C2V vertices
    # (0, 1: edge endpoints; 2, 3: far vertices of the adjacent cells).
    vn_vert = u_vert(E2C2V) * primal_normal_vert_x + v_vert(E2C2V) * primal_normal_vert_y
    vt_vert = u_vert(E2C2V) * dual_normal_vert_x + v_vert(E2C2V) * dual_normal_vert_y

    # Vertical velocity at full levels: cell centers (E2C) and edge endpoints (E2C2V 0, 1).
    w_full_c1 = wpfloat("0.5") * (w(E2C[0]) + w(E2C[0])(KDim + 1))
    w_full_c2 = wpfloat("0.5") * (w(E2C[1]) + w(E2C[1])(KDim + 1))
    w_full_v1 = wpfloat("0.5") * (w_vert(E2C2V[0]) + w_vert(E2C2V[0])(KDim + 1))
    w_full_v2 = wpfloat("0.5") * (w_vert(E2C2V[1]) + w_vert(E2C2V[1])(KDim + 1))

    # Velocity gradient tensor at edge of full levels, e.g. T_12 = du_1/dx_2.
    vgrad_11 = (vn_vert[E2C2VDim(3)] - vn_vert[E2C2VDim(2)]) * inv_vert_vert_length
    vgrad_12 = (
        (vn_vert[E2C2VDim(1)] - vn_vert[E2C2VDim(0)]) * tangent_orientation * inv_primal_edge_length
    )
    vgrad_13 = (vn_ie - vn_ie(KDim + 1)) * inv_ddqz_z_full_e

    vgrad_21 = (vt_vert[E2C2VDim(3)] - vt_vert[E2C2VDim(2)]) * inv_vert_vert_length
    vgrad_22 = (
        (vt_vert[E2C2VDim(1)] - vt_vert[E2C2VDim(0)]) * tangent_orientation * inv_primal_edge_length
    )
    vgrad_23 = (vt_ie - vt_ie(KDim + 1)) * inv_ddqz_z_full_e

    vgrad_31 = (w_full_c2 - w_full_c1) * inv_dual_edge_length
    vgrad_32 = (w_full_v2 - w_full_v1) * tangent_orientation * inv_primal_edge_length
    vgrad_33 = (w_ie - w_ie(KDim + 1)) * inv_ddqz_z_full_e

    # Strain rates at edge center, D_ij = 2 * S_ij = du_i/dx_j + du_j/dx_i.
    d_12 = vgrad_12 + vgrad_21
    d_13 = vgrad_13 + vgrad_31
    d_23 = vgrad_23 + vgrad_32

    # shear = 2 * |S|^2 with |S| = sqrt(2 * S_ij * S_ij);
    # mechanical production is half of this value multiplied by km.
    shear = wpfloat("4.0") * (
        vgrad_11 * vgrad_11 + vgrad_22 * vgrad_22 + vgrad_33 * vgrad_33
    ) + wpfloat("2.0") * (d_12 * d_12 + d_13 * d_13 + d_23 * d_23)

    # Trace of the strain-rate tensor S_ij: trace(S_ij) = S_jj = 0.5 * D_jj = du_j/dx_j.
    div_stress = vgrad_11 + vgrad_22 + vgrad_33

    return shear, div_stress


@gtx.field_operator
def _compute_edge_shear_diagnostics(
    w: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    wgtfac_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e: fa.EdgeKField[wpfloat],
    wgtfacq_e: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    """
    Edge diagnostics of ``Compute_diagnostics`` (mo_vdf_atmo.f90).

    Fuses ``cells2edges_scalar`` (w -> w_ie),
    ``interpolate_normal_velocity_edge_interface`` (vn -> vn_ie),
    ``rbf_vec_interpol_edge`` (vn_ie -> vt_ie) and the shear/divergence of the
    stress tensor. ``w_ie`` only depends on the (already valid) input ``w``, so
    moving it from before the vertex exchange to this group does not change the
    results; it is the field the shear needs.

    Returns:
        vertical velocity, normal and tangential velocity at half-level edges,
        and shear and divergence of the stress at full-level edges
    """
    w_ie = _cell_2_edge_interpolation(w, c_lin_e)
    vn_ie = _interpolate_edge_field_to_half_levels_with_boundaries_wp(
        interpolant=vn,
        wgtfac_e=wgtfac_e,
        wgtfacq1_e=wgtfacq1_e,
        wgtfacq_e=wgtfacq_e,
        nlev=nlev,
    )
    vt_ie = _compute_tangential_wind_wp(vn=vn_ie, rbf_vec_coeff_e=rbf_vec_coeff_e)
    shear, div_stress = _compute_shear_and_div_of_stress(
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        w=w,
        vn_ie=vn_ie,
        vt_ie=vt_ie,
        w_ie=w_ie,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
    )
    return w_ie, vn_ie, vt_ie, shear, div_stress


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_shear_diagnostics(
    w: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    wgtfac_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e: fa.EdgeKField[wpfloat],
    wgtfacq_e: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    w_ie: fa.EdgeKField[wpfloat],
    vn_ie: fa.EdgeKField[wpfloat],
    vt_ie: fa.EdgeKField[wpfloat],
    shear: fa.EdgeKField[wpfloat],
    div_stress: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
    edge_start_lateral_boundary_level_2: gtx.int32,
    edge_start_lateral_boundary_level_3: gtx.int32,
    edge_start_lateral_boundary_level_4: gtx.int32,
    edge_end_halo_level_2: gtx.int32,
    edge_end_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
    vertical_end_half: gtx.int32,
) -> None:
    _compute_edge_shear_diagnostics(
        w=w,
        vn=vn,
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        c_lin_e=c_lin_e,
        wgtfac_e=wgtfac_e,
        wgtfacq1_e=wgtfacq1_e,
        wgtfacq_e=wgtfacq_e,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        nlev=nlev,
        out=(w_ie, vn_ie, vt_ie, shear, div_stress),
        domain=(
            # w_ie: edges rl 2..min_rledge_int-2, all half levels
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_2, edge_end_halo_level_2),
                dims.KDim: (vertical_start, vertical_end_half),
            },
            # vn_ie: edges rl 2..min_rledge_int-3, all half levels. There is no
            # icon4py zone for min_rledge_int-3 (third halo line); h_grid.Zone.END
            # is the closest more-inclusive bound (identical on a single node,
            # where there are no halo lines; the extra halo rows are unused
            # boundary values anyway).
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_2, edge_end_end),
                dims.KDim: (vertical_start, vertical_end_half),
            },
            # vt_ie: edges rl 3..min_rledge_int-2, all half levels
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_3, edge_end_halo_level_2),
                dims.KDim: (vertical_start, vertical_end_half),
            },
            # shear / div_stress: edges rl 4..min_rledge_int-2, all full levels
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_4, edge_end_halo_level_2),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_4, edge_end_halo_level_2),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics, after the vertex exchange: cell diagnostics
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_strain_rate_diagnostics(
    shear: fa.EdgeKField[wpfloat],
    div_stress: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Cell-centered strain-rate diagnostics of ``Compute_diagnostics``
    (mo_vdf_atmo.f90): the two C2E gathers of the edge stress diagnostics.

    The mechanical production term is the port of
    ``interpolate_rate_of_strain_full2half_edge2cell``; its Fortran loop runs
    over jk = 2..nlev (1-based), i.e. half levels k = 1..nlev-1 (0-based), so
    the top (k = 0) and bottom (k = nlev) rows are not computed.

    Returns:
        divergence of the stress at full-level cells and the mechanical
        production term at half-level cells
    """
    div_c = _interpolate_to_cell_center_wp(interpolant=div_stress, e_bln_c_s=e_bln_c_s)
    mech_prod = _interpolate_edge_field_to_cell_half_levels_wp(
        interpolant=shear, e_bln_c_s=e_bln_c_s, wgtfac_c=wgtfac_c
    )
    return div_c, mech_prod


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_strain_rate_diagnostics(
    shear: fa.EdgeKField[wpfloat],
    div_stress: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    mech_prod: fa.CellKField[wpfloat],
    cell_start_nudging: gtx.int32,
    cell_start_lateral_boundary_level_3: gtx.int32,
    cell_end_halo: gtx.int32,
    vertical_start: gtx.int32,
    vertical_start_interior: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_strain_rate_diagnostics(
        shear=shear,
        div_stress=div_stress,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        out=(div_c, mech_prod),
        domain=(
            # div_c: cells rl grf_bdywidth_c+1..min_rlcell_int-1, all full levels
            {
                dims.CellDim: (cell_start_nudging, cell_end_halo),
                dims.KDim: (vertical_start, vertical_end),
            },
            # mech_prod: cells rl 3..min_rlcell_int-1, half levels 1..nlev-1
            {
                dims.CellDim: (cell_start_lateral_boundary_level_3, cell_end_halo),
                dims.KDim: (vertical_start_interior, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics: eddy viscosity and diffusivity
# ---------------------------------------------------------------------------
@gtx.field_operator
def _stability_term_classic(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    rturb_prandtl: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the classic (Lilly 1962) stability correction term for the eddy viscosity:

        stability_term = sqrt(max(0, |S|^2 - N^2 / Pr_t))

    with |S|^2 = 0.5 * mech_prod the square of the strain rate magnitude,
    N^2 = bruvais the Brunt-Vaisala frequency squared, and
    1 / Pr_t = rturb_prandtl the reciprocal turbulent Prandtl number.
    """
    return sqrt(maximum(wpfloat("0.0"), wpfloat("0.5") * mech_prod - rturb_prandtl * bruvais))


@gtx.field_operator
def _stability_term_louis(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the stability correction term for the eddy viscosity based on the
    stability correction function of Louis (1979):

        Ri = 2 * N^2 / max(eps, mech_prod)
        stability_function = max(1 - Ri / Pr_t,
                                 min(1, (1 / (1 + b * scaling * |Ri|))^4))
        stability_term = sqrt(0.5 * mech_prod * stability_function)
    """
    # Threshold to avoid division by zero in the Richardson number (``eps_louis``
    # in ICON's mo_tmx_smagorinsky.f90). Defined here because module-level closure
    # constants are not supported by the gtfn backend.
    eps_louis = wpfloat("1.0e-28")
    ri = wpfloat("2.0") * bruvais / maximum(eps_louis, mech_prod)

    stability_function = maximum(
        wpfloat("1.0") - ri * rturb_prandtl,
        minimum(
            wpfloat("1.0"),
            power(
                wpfloat("1.0")
                / (wpfloat("1.0") + louis_constant_b * scaling_factor_louis * abs(ri)),
                wpfloat("4.0"),
            ),
        ),
    )

    return sqrt(wpfloat("0.5") * mech_prod * stability_function)


@gtx.field_operator
def _compute_smagorinsky_viscosity(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    mixing_length_sq: fa.CellKField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the eddy viscosity and diffusivity at half-level cell centers based on
    the Smagorinsky-Lilly eddy viscosity model.

    Port of ``Smagorinsky_model`` in ICON's ``mo_tmx_smagorinsky.f90``:
    - interior half levels (0 < k < nlev):
        km_ic = rho_ic * mixing_length_sq * stability_term
        kh_ic = km_ic * rturb_prandtl
    - boundary half levels are copies of the adjacent interior rows:
        k = 0 copies k = 1, k = nlev copies k = nlev - 1
      (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev).

    Depending on the configuration, the classic (Lilly 1962) or the Louis (1979)
    stability correction function is used. If the Louis formulation is enabled but
    excluded over land (``use_louis_land = False``) and/or sea ice
    (``use_louis_ice = False``), cells with more than 50% land fraction and/or more
    than 50% ice fraction fall back to the classic formulation.

    ``use_louis``, ``use_louis_land`` and ``use_louis_ice`` are scalar configuration
    flags; they can be passed as static (compile-time) arguments so that only the
    selected variant is compiled.
    """
    if use_louis:
        stability_classic = _stability_term_classic(
            mech_prod=mech_prod, bruvais=bruvais, rturb_prandtl=rturb_prandtl
        )
        stability_louis = _stability_term_louis(
            mech_prod=mech_prod,
            bruvais=bruvais,
            scaling_factor_louis=scaling_factor_louis,
            rturb_prandtl=rturb_prandtl,
            louis_constant_b=louis_constant_b,
        )
        if use_louis_land:
            if use_louis_ice:
                stability_term = stability_louis
            else:
                stability_term = where(
                    fract_ice > wpfloat("0.5"), stability_classic, stability_louis
                )
        else:
            if use_louis_ice:
                stability_term = where(
                    fract_land > wpfloat("0.5"), stability_classic, stability_louis
                )
            else:
                stability_term = where(
                    (fract_land > wpfloat("0.5")) | (fract_ice > wpfloat("0.5")),
                    stability_classic,
                    stability_louis,
                )
    else:
        stability_term = _stability_term_classic(
            mech_prod=mech_prod, bruvais=bruvais, rturb_prandtl=rturb_prandtl
        )

    km = rho_ic * mixing_length_sq * stability_term
    km_ic = concat_where(dims.KDim == 0, km(KDim + 1), km)
    km_ic = concat_where(dims.KDim == nlev, km(KDim - 1), km_ic)
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_smagorinsky_viscosity(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    mixing_length_sq: fa.CellKField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_smagorinsky_viscosity(
        mech_prod=mech_prod,
        bruvais=bruvais,
        rho_ic=rho_ic,
        mixing_length_sq=mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        rturb_prandtl=rturb_prandtl,
        louis_constant_b=louis_constant_b,
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
        nlev=nlev,
        out=(km_ic, kh_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _assign_constant_viscosity(
    rho_ic: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Assign a constant eddy viscosity and diffusivity (for turbulence model validation).

    Port of ``Assign_constant_eddy_viscosity`` in ICON's ``mo_vdf_atmo.f90``:
    - interior half levels (0 < k < nlev):
        km_ic = rho_ic * km_const
        kh_ic = km_ic * rturb_prandtl
    - boundary half levels are copies of the adjacent interior rows:
        k = 0 copies k = 1, k = nlev copies k = nlev - 1
      (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev).

    Args:
        rho_ic: air density at half-level cell centers (nlev + 1 levels)
        km_const: constant kinematic eddy viscosity
        rturb_prandtl: reciprocal turbulent Prandtl number
        nlev: number of full levels

    Returns:
        eddy viscosity km_ic and eddy diffusivity kh_ic at half levels
    """
    km = rho_ic * km_const
    km_ic = concat_where(dims.KDim == 0, km(KDim + 1), km)
    km_ic = concat_where(dims.KDim == nlev, km(KDim - 1), km_ic)
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def assign_constant_viscosity(
    rho_ic: fa.CellKField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _assign_constant_viscosity(
        rho_ic=rho_ic,
        km_const=km_const,
        rturb_prandtl=rturb_prandtl,
        nlev=nlev,
        out=(km_ic, kh_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics, after the km_ic/kh_ic cell exchange: one gather per
# horizontal dimension, so nothing can be fused here
# ---------------------------------------------------------------------------
@gtx.field_operator
def _interpolate_km_to_full_level_cells(
    km_ic: fa.CellKField[wpfloat],
    km_min: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Interpolate the eddy viscosity from half-level cell centers to full-level
    cell centers and apply the minimum-viscosity floor:

        km_c(k) = max(km_min, 0.5 * (km_ic(k) + km_ic(k + 1)))

    Port of ``interpolate_eddy_viscosity2cell`` in ICON's ``mo_vdf_atmo.f90``.
    Domains (Fortran call site in ``Compute_diagnostics``): jk = 1..nlev;
    ``rl_start = grf_bdywidth_c`` -> ``h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4``,
    ``rl_end = min_rlcell_int - 1`` -> ``h_grid.Zone.HALO`` (halo cells are
    computed on purpose because ``km_c`` is used in the diffusion later).
    The floor deliberately lives here (and in the vertex/edge interpolations)
    and not in the Smagorinsky viscosity computation, matching the Fortran.
    """
    return maximum(km_min, average_level_plus1_on_cells(km_ic))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km_to_full_level_cells(
    km_ic: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    km_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_km_to_full_level_cells(
        km_ic=km_ic,
        km_min=km_min,
        out=km_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _interpolate_km_to_vertices(
    km_ic: fa.CellKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    km_min: wpfloat,
) -> fa.VertexKField[wpfloat]:
    """
    Interpolate the eddy viscosity from half-level cell centers to half-level
    vertices (area-weighted V2C gather with ``cells_aw_verts``) and apply the
    minimum-viscosity floor:

        km_iv = max(km_min, sum_{c in V2C} cells_aw_verts * km_ic(c))

    Port of ``interpolate_eddy_viscosity2half_vertex`` in ICON's
    ``mo_vdf_atmo.f90``. Domains: all half levels (``cells2verts_scalar``
    defaults to the full column); ``opt_rlstart = 5 (= max_rlvert)`` ->
    ``h_grid.Zone.NUDGING``, ``opt_rlend = min_rlvert_int - 1`` ->
    ``h_grid.Zone.HALO`` (halo vertices are computed on purpose).

    Note: the Fortran applies the ``MAX(km_min, ...)`` floor to the *entire*
    ``km_iv`` array (which was initialized to zero beforehand), so vertices
    outside the interpolated region end up holding ``km_min``. Here the floor
    is fused with the gather and only acts on the program domain; the caller
    must initialize ``km_iv`` to ``km_min`` (instead of zero) if values outside
    this domain are ever read.
    """
    return maximum(km_min, _compute_cell_2_vertex_interpolation(km_ic, cells_aw_verts))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km_to_vertices(
    km_ic: fa.CellKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    km_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_km_to_vertices(
        km_ic=km_ic,
        cells_aw_verts=cells_aw_verts,
        km_min=km_min,
        out=km_iv,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _interpolate_km_to_edges(
    km_ic: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    km_min: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """
    Interpolate the eddy viscosity from half-level cell centers to half-level
    edges (linear E2C gather with ``c_lin_e``) and apply the minimum-viscosity
    floor:

        km_ie = max(km_min, sum_{c in E2C} c_lin_e * km_ic(c))

    Port of ``interpolate_eddy_viscosity2half_edge`` in ICON's
    ``mo_vdf_atmo.f90``. The single-neighbor lateral-boundary fill of
    ``cells2edges_scalar`` (edges with ``refin_ctrl`` 1..2) is not reached at
    this call site (``opt_rlstart = grf_bdywidth_e``) and is not ported.
    Domains: all half levels; ``opt_rlstart = grf_bdywidth_e (= 9)`` ->
    ``h_grid.Zone.NUDGING``, ``opt_rlend = min_rledge_int - 1`` ->
    ``h_grid.Zone.HALO`` (halo edges are computed on purpose).

    Note: the Fortran applies the ``MAX(km_min, ...)`` floor to the *entire*
    ``km_ie`` array (which was initialized to zero beforehand), so edges outside
    the interpolated region end up holding ``km_min``. Here the floor is fused
    with the gather and only acts on the program domain; the caller must
    initialize ``km_ie`` to ``km_min`` (instead of zero) if values outside this
    domain are ever read.
    """
    return maximum(km_min, _cell_2_edge_interpolation(km_ic, c_lin_e))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km_to_edges(
    km_ic: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    km_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_km_to_edges(
        km_ic=km_ic,
        c_lin_e=c_lin_e,
        km_min=km_min,
        out=km_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ---------------------------------------------------------------------------
# Update_diagnostics (mo_vdf_atmo.f90 l. 487 and mo_vdf.f90 l. 354)
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_vertical_integral_diagnostics(
    dry_static_energy: fa.CellKField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    dz: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Compute the running vertical integrals of the tmx energy diagnostics.

    Port of the vertical-integral part of 'Update_diagnostics' in
    mo_vdf_atmo.f90 ('compute_internal_energy_vi' and the accumulation loop):

        ctgzvi             = sum_k ctgz(k) * rho(k) * dz(k)
        dissip_ke_vi       = sum_k dissip_ke(k)
        int_energy_vi      = sum_k internal_energy(new state, k)
        int_energy_vi_tend = (int_energy_vi - sum_k internal_energy(old state, k))
                             / dtime

    with ``internal_energy`` from mo_aes_thermo.f90 (ported in
    :mod:`icon4py.model.common.physics.thermodynamics`; the liquid phase is
    qc + qr, the solid phase qi + qs + qg). qr, qs and qg are not diffused
    and have no new state. The Fortran diagnostics are 2D surface fields;
    here each output holds the running top-down sum, so its value at the last
    full level (k = nlev - 1) is the column integral the caller extracts.
    """
    int_energy_old = _internal_energy(
        t=temperature, qv=qv, qliq=qc + qr, qice=qi + qs + qg, rho=rho, dz=dz
    )
    int_energy_new = _internal_energy(
        t=new_temperature,
        qv=new_qv,
        qliq=new_qc + qr,
        qice=new_qi + qs + qg,
        rho=rho,
        dz=dz,
    )
    cptgz_vi = _compute_vertical_integral(dry_static_energy * rho * dz)
    dissip_ke_vi = _compute_vertical_integral(dissip_ke)
    int_energy_vi = _compute_vertical_integral(int_energy_new)
    int_energy_vi_old = _compute_vertical_integral(int_energy_old)
    int_energy_vi_tend = (int_energy_vi - int_energy_vi_old) / dtime
    return cptgz_vi, dissip_ke_vi, int_energy_vi, int_energy_vi_tend


@gtx.field_operator
def _update_exchange_coefficient_diagnostics(
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    use_km_const: bool,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Assemble the full-level exchange coefficient diagnostics ``km`` / ``kh``.

    Port of the km/kh loop of 'Update_diagnostics' in mo_vdf.f90 (these fields
    are output-only diagnostics, nothing in tmx reads them):

        km(k) = km_ic(k + 1),  kh(k) = kh_ic(k + 1)     for jk = 1..nlev-1
        km(nlev) = km_const,   kh(nlev) = km_const * rturb_prandtl
                                                        if use_km_const
        km(nlev) = km_sfc,     kh(nlev) = kh_sfc        otherwise

    The surface exchange coefficients ``km_sfc`` / ``kh_sfc`` are aggregated
    from the surface tiles (mo_vdf_diag_smag.f90) and are out of scope of the
    atmosphere-only port: the bottom row is set to zero when ``use_km_const``
    is False. See the module docstring for the ``KDim < nlev - 1`` selector and
    the anchored constant branches.
    """
    shifted_km = km_ic(KDim + 1)
    shifted_kh = kh_ic(KDim + 1)
    if use_km_const:
        km_bottom = shifted_km * wpfloat("0.0") + km_const
        kh_bottom = shifted_kh * wpfloat("0.0") + km_const * rturb_prandtl
    else:
        km_bottom = shifted_km * wpfloat("0.0")
        kh_bottom = shifted_kh * wpfloat("0.0")
    km = concat_where(dims.KDim < nlev - 1, shifted_km, km_bottom)
    kh = concat_where(dims.KDim < nlev - 1, shifted_kh, kh_bottom)
    return km, kh


@gtx.field_operator
def _update_end_of_step_diagnostics(
    new_temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    dz: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    use_km_const: bool,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    End-of-step diagnostics of ``Update_diagnostics``.

    Fuses the recomputation of the dry static energy from the updated
    temperature, the vertical-integral diagnostics that consume it, and the
    full-level exchange coefficient diagnostics.

    Returns:
        dry static energy, its running vertical integral, the running vertical
        integrals of the kinetic energy dissipation, of the internal energy and
        of the internal energy tendency, and the full-level eddy viscosity and
        diffusivity
    """
    dry_static_energy = _compute_dry_static_energy(
        temperature=new_temperature, height_above_ground=height_above_ground, grav=grav
    )
    cptgz_vi, dissip_ke_vi, int_energy_vi, int_energy_vi_tend = (
        _compute_vertical_integral_diagnostics(
            dry_static_energy=dry_static_energy,
            dissip_ke=dissip_ke,
            rho=rho,
            dz=dz,
            temperature=temperature,
            qv=qv,
            qc=qc,
            qi=qi,
            new_temperature=new_temperature,
            new_qv=new_qv,
            new_qc=new_qc,
            new_qi=new_qi,
            qr=qr,
            qs=qs,
            qg=qg,
            dtime=dtime,
        )
    )
    km, kh = _update_exchange_coefficient_diagnostics(
        km_ic=km_ic,
        kh_ic=kh_ic,
        km_const=km_const,
        rturb_prandtl=rturb_prandtl,
        use_km_const=use_km_const,
        nlev=nlev,
    )
    return dry_static_energy, cptgz_vi, dissip_ke_vi, int_energy_vi, int_energy_vi_tend, km, kh


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_end_of_step_diagnostics(
    new_temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    dz: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    dry_static_energy: fa.CellKField[wpfloat],
    cptgz_vi: fa.CellKField[wpfloat],
    dissip_ke_vi: fa.CellKField[wpfloat],
    int_energy_vi: fa.CellKField[wpfloat],
    int_energy_vi_tend: fa.CellKField[wpfloat],
    km: fa.CellKField[wpfloat],
    kh: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    use_km_const: bool,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_end_of_step_diagnostics(
        new_temperature=new_temperature,
        height_above_ground=height_above_ground,
        dissip_ke=dissip_ke,
        rho=rho,
        dz=dz,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        new_qv=new_qv,
        new_qc=new_qc,
        new_qi=new_qi,
        qr=qr,
        qs=qs,
        qg=qg,
        km_ic=km_ic,
        kh_ic=kh_ic,
        grav=grav,
        dtime=dtime,
        km_const=km_const,
        rturb_prandtl=rturb_prandtl,
        use_km_const=use_km_const,
        nlev=nlev,
        out=(
            dry_static_energy,
            cptgz_vi,
            dissip_ke_vi,
            int_energy_vi,
            int_energy_vi_tend,
            km,
            kh,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

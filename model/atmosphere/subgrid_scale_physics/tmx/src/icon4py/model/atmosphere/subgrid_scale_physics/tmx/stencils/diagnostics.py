# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stencils of the tmx diagnostics.

Ports ``Smagorinsky_init`` (mo_tmx_smagorinsky.f90) and ``Compute_diagnostics``
(mo_vdf_atmo.f90).
"""

import gt4py.next as gtx
from gt4py.next import abs, maximum, minimum, power, sqrt, where  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import E2C, E2C2V, E2C2VDim, KDim
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation_on_half_levels,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_on_half_levels,
)
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels import (
    _interpolate_cell_field_to_half_levels_with_boundaries,
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels import (
    _interpolate_edge_field_to_half_levels_with_boundaries,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.math.tensor_operations import (
    Mat3OnEdges,
    squared_norm_of_symmetric_on_edges,
    trace_on_edges,
    twice_symmetric_part_on_edges,
)
from icon4py.model.common.math.vertical_operations import (
    average_level_plus1_on_cells,
    with_boundaries_on_half_levels_on_cells,
)
from icon4py.model.common.physics.compute_brunt_vaisala_frequency import (
    _compute_brunt_vaisala_frequency,
)
from icon4py.model.common.physics.thermodynamics.compute_energy import _compute_dry_static_energy
from icon4py.model.common.physics.thermodynamics.compute_temperature import (
    _compute_virtual_potential_temperature,
)
from icon4py.model.common.type_alias import wpfloat


# ---------------------------------------------------------------------------
# Smagorinsky_init (mo_tmx_smagorinsky.f90): run once, at granule construction
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_smagorinsky_mixing_length(
    ddqz_z_half: fa.CellKHalfField[wpfloat],
    geopot_agl_ifc: fa.CellKHalfField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    smag_constant: wpfloat,
    max_turb_scale: wpfloat,
    grav: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    """
    Compute the square of the subgrid-scale mixing length for the Smagorinsky model.

    Port of ``compute_mixing_length`` in ICON's ``mo_tmx_smagorinsky.f90``:

        lambda^2 = (Cs * Delta)^2 * (kappa * x_3)^2 / ((Cs * Delta)^2 + (kappa * x_3)^2)
                 = (Cs * Delta * x_3)^2 / ((Cs * Delta / kappa)^2 + x_3^2)

    with Cs the Smagorinsky constant, Delta the filter/grid width (capped at
    ``max_turb_scale``), x_3 the height above ground, and kappa = 0.4 the
    von Karman constant. Reference: Dipankar et al. (2015).

    Args:
        ddqz_z_half: layer thickness centered at half levels
        geopot_agl_ifc: geopotential above ground at half levels
        cell_area: cell area
        smag_constant: Smagorinsky constant Cs
        max_turb_scale: maximum turbulence length scale
        grav: gravitational acceleration

    Returns:
        square of the Smagorinsky mixing length at half levels
    """
    kappa = PhysicsConstants.von_karman

    z_agl = geopot_agl_ifc * (wpfloat("1.0") / grav)
    les_filter = smag_constant * minimum(
        max_turb_scale, power(ddqz_z_half * cell_area, wpfloat("0.33333"))
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
    # global mean cell area of the R2B8 grid [m^2] (``mean_area_R2B8`` in mo_tmx_smagorinsky.f90)
    mean_cell_area_r2b8 = wpfloat("97294071.23714285")
    return mean_cell_area_r2b8 / cell_area


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_smagorinsky_mixing_length(
    ddqz_z_half: fa.CellKHalfField[wpfloat],
    geopot_agl_ifc: fa.CellKHalfField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    mixing_length_sq: fa.CellKHalfField[wpfloat],
    smag_constant: wpfloat,
    max_turb_scale: wpfloat,
    grav: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_smagorinsky_mixing_length(
        ddqz_z_half=ddqz_z_half,
        geopot_agl_ifc=geopot_agl_ifc,
        cell_area=cell_area,
        smag_constant=smag_constant,
        max_turb_scale=max_turb_scale,
        grav=grav,
        out=mixing_length_sq,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
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
# Compute_diagnostics: thermodynamic cell diagnostics
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_thermodynamic_diagnostics(
    temperature: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    grav: wpfloat,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKHalfField[wpfloat],
]:
    """
    Thermodynamic cell diagnostics of ``Compute_diagnostics`` (mo_vdf_atmo.f90).

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
    rho_ic = _interpolate_cell_field_to_half_levels_with_boundaries(
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
    wgtfac_c: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    dry_static_energy: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rho_ic: fa.CellKHalfField[wpfloat],
    bruvais: fa.CellKHalfField[wpfloat],
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
            # dry_static_energy
            {
                dims.CellDim: (cell_start_nudging, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            # theta_v
            {
                dims.CellDim: (cell_start_lateral_boundary_level_3, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            # rho_ic
            {
                dims.CellDim: (cell_start_lateral_boundary_level_2, cell_end_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
            # bruvais
            {
                dims.CellDim: (cell_start_lateral_boundary_level_3, cell_end_local),
                dims.KHalfDim: (vertical_start_interior, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics: wind at vertices
# ---------------------------------------------------------------------------
@gtx.field_operator
def _interpolate_wind_to_vertices(
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
) -> tuple[fa.VertexKHalfField[wpfloat], fa.VertexKField[wpfloat], fa.VertexKField[wpfloat]]:
    """
    Interpolate the wind components to the vertices.

    The vertical component is averaged from the V2C neighbor cells with the
    weights ``cells_aw_verts``; the horizontal components are reconstructed
    from the edge-normal component with the RBF coefficients ``rbf_coeff_v1``
    and ``rbf_coeff_v2``.

    Note that ``w`` lives on half levels and ``vn`` on full levels, so the
    three outputs do not share a vertical domain.
    """
    w_vert = _compute_cell_2_vertex_interpolation(w, cells_aw_verts)
    u_vert, v_vert = _mo_intp_rbf_rbf_vec_interpol_vertex(
        p_e_in=vn, ptr_coeff_1=rbf_coeff_v1, ptr_coeff_2=rbf_coeff_v2
    )
    return w_vert, u_vert, v_vert


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_wind_to_vertices(
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
    vertical_end_half: gtx.int32,
) -> None:
    _interpolate_wind_to_vertices(
        w=w,
        vn=vn,
        cells_aw_verts=cells_aw_verts,
        rbf_coeff_v1=rbf_coeff_v1,
        rbf_coeff_v2=rbf_coeff_v2,
        out=(w_vert, u_vert, v_vert),
        domain=(
            # w_vert: half levels
            {
                dims.VertexDim: (horizontal_start, horizontal_end),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
            # u_vert / v_vert: full levels
            {
                dims.VertexDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.VertexDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics: edge diagnostics
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_velocity_gradient_tensor(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    vn_ie: fa.EdgeKHalfField[wpfloat],
    vt_ie: fa.EdgeKHalfField[wpfloat],
    w_ie: fa.EdgeKHalfField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
) -> Mat3OnEdges:
    """
    The velocity gradient tensor ``T_ij = du_i/dx_j`` at edges of full levels.

    Port of ``compute_velocity_gradient_tensor`` (mo_vdf_atmo.f90). The first index is
    the velocity component, the second the derivative direction; 0: edge-normal,
    1: edge-tangential, 2: vertical.
    """
    # Normal/tangential velocity components at the four E2C2V vertices
    # (0, 1: edge endpoints; 2, 3: far vertices of the adjacent cells).
    vn_vert = u_vert(E2C2V) * primal_normal_vert_x + v_vert(E2C2V) * primal_normal_vert_y
    vt_vert = u_vert(E2C2V) * dual_normal_vert_x + v_vert(E2C2V) * dual_normal_vert_y

    # Vertical velocity at full levels: cell centers (E2C) and edge endpoints (E2C2V 0, 1).
    w_full_c1 = wpfloat("0.5") * (w(E2C[0])(KDim - 0.5) + w(E2C[0])(KDim + 0.5))
    w_full_c2 = wpfloat("0.5") * (w(E2C[1])(KDim - 0.5) + w(E2C[1])(KDim + 0.5))
    w_full_v1 = wpfloat("0.5") * (w_vert(E2C2V[0])(KDim - 0.5) + w_vert(E2C2V[0])(KDim + 0.5))
    w_full_v2 = wpfloat("0.5") * (w_vert(E2C2V[1])(KDim - 0.5) + w_vert(E2C2V[1])(KDim + 0.5))

    return (
        (
            (vn_vert[E2C2VDim(3)] - vn_vert[E2C2VDim(2)]) * inv_vert_vert_length,
            (vn_vert[E2C2VDim(1)] - vn_vert[E2C2VDim(0)])
            * tangent_orientation
            * inv_primal_edge_length,
            (vn_ie(KDim - 0.5) - vn_ie(KDim + 0.5)) * inv_ddqz_z_full_e,
        ),
        (
            (vt_vert[E2C2VDim(3)] - vt_vert[E2C2VDim(2)]) * inv_vert_vert_length,
            (vt_vert[E2C2VDim(1)] - vt_vert[E2C2VDim(0)])
            * tangent_orientation
            * inv_primal_edge_length,
            (vt_ie(KDim - 0.5) - vt_ie(KDim + 0.5)) * inv_ddqz_z_full_e,
        ),
        (
            (w_full_c2 - w_full_c1) * inv_dual_edge_length,
            (w_full_v2 - w_full_v1) * tangent_orientation * inv_primal_edge_length,
            (w_ie(KDim - 0.5) - w_ie(KDim + 0.5)) * inv_ddqz_z_full_e,
        ),
    )


@gtx.field_operator
def _compute_shear_and_div_of_stress(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    vn_ie: fa.EdgeKHalfField[wpfloat],
    vt_ie: fa.EdgeKHalfField[wpfloat],
    w_ie: fa.EdgeKHalfField[wpfloat],
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

        shear         = ||D||^2 = 2 |S|^2,  D = T + T^t = 2 S
        div_of_stress = trace(T)

    with T the velocity gradient and S the strain rate. Mechanical production is
    half of ``shear`` multiplied by km.
    """
    velocity_gradient = _compute_velocity_gradient_tensor(
        u_vert,
        v_vert,
        w_vert,
        w,
        vn_ie,
        vt_ie,
        w_ie,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        inv_dual_edge_length,
        inv_ddqz_z_full_e,
    )
    strain_rate = twice_symmetric_part_on_edges(velocity_gradient)  # D = T + T^t = 2 S
    shear = squared_norm_of_symmetric_on_edges(strain_rate)
    div_of_stress = trace_on_edges(velocity_gradient)

    return shear, div_of_stress


@gtx.field_operator
def _compute_edge_shear_diagnostics(
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    wgtfac_e: fa.EdgeKHalfField[wpfloat],
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
    fa.EdgeKHalfField[wpfloat],
    fa.EdgeKHalfField[wpfloat],
    fa.EdgeKHalfField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    """
    Edge diagnostics of ``Compute_diagnostics`` (mo_vdf_atmo.f90).

    Returns:
        vertical velocity, normal and tangential velocity at half-level edges,
        and shear and divergence of the stress at full-level edges
    """
    w_ie = _cell_2_edge_interpolation_on_half_levels(w, c_lin_e)
    vn_ie = _interpolate_edge_field_to_half_levels_with_boundaries(
        interpolant=vn,
        wgtfac_e=wgtfac_e,
        wgtfacq1_e=wgtfacq1_e,
        wgtfacq_e=wgtfacq_e,
        nlev=nlev,
    )
    vt_ie = _compute_tangential_wind_on_half_levels(vn=vn_ie, rbf_vec_coeff_e=rbf_vec_coeff_e)
    shear, div_of_stress = _compute_shear_and_div_of_stress(
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
    return w_ie, vn_ie, vt_ie, shear, div_of_stress


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_shear_diagnostics(
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    wgtfac_e: fa.EdgeKHalfField[wpfloat],
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
    w_ie: fa.EdgeKHalfField[wpfloat],
    vn_ie: fa.EdgeKHalfField[wpfloat],
    vt_ie: fa.EdgeKHalfField[wpfloat],
    shear: fa.EdgeKField[wpfloat],
    div_of_stress: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
    edge_start_lateral_boundary_level_2: gtx.int32,
    edge_start_lateral_boundary_level_3: gtx.int32,
    edge_start_lateral_boundary_level_4: gtx.int32,
    edge_end_halo_level_2: gtx.int32,
    edge_end_halo_level_3: gtx.int32,
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
        out=(w_ie, vn_ie, vt_ie, shear, div_of_stress),
        domain=(
            # w_ie
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_2, edge_end_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
            # vn_ie
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_2, edge_end_halo_level_3),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
            # vt_ie
            {
                dims.EdgeDim: (edge_start_lateral_boundary_level_3, edge_end_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
            # shear / div_of_stress
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
# Compute_diagnostics: cell diagnostics
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_strain_rate_diagnostics(
    shear: fa.EdgeKField[wpfloat],
    div_of_stress: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKHalfField[wpfloat]]:
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
    div_c = _interpolate_to_cell_center(interpolant=div_of_stress, e_bln_c_s=e_bln_c_s)
    mech_prod = _interpolate_cell_field_to_half_levels_wp(
        wgtfac_c=wgtfac_c,
        interpolant=_interpolate_to_cell_center(interpolant=shear, e_bln_c_s=e_bln_c_s),
    )
    return div_c, mech_prod


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_strain_rate_diagnostics(
    shear: fa.EdgeKField[wpfloat],
    div_of_stress: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    mech_prod: fa.CellKHalfField[wpfloat],
    cell_start_nudging: gtx.int32,
    cell_start_lateral_boundary_level_3: gtx.int32,
    cell_end_halo: gtx.int32,
    vertical_start: gtx.int32,
    vertical_start_interior: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_strain_rate_diagnostics(
        shear=shear,
        div_of_stress=div_of_stress,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        out=(div_c, mech_prod),
        domain=(
            # div_c
            {
                dims.CellDim: (cell_start_nudging, cell_end_halo),
                dims.KDim: (vertical_start, vertical_end),
            },
            # mech_prod
            {
                dims.CellDim: (cell_start_lateral_boundary_level_3, cell_end_halo),
                dims.KHalfDim: (vertical_start_interior, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics: eddy viscosity and diffusivity
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_stability_term_classic(
    mech_prod: fa.CellKHalfField[wpfloat],
    bruvais: fa.CellKHalfField[wpfloat],
    rturb_prandtl: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    """
    Compute the classic (Lilly 1962) stability correction term for the eddy viscosity:

        stability_term = sqrt(max(0, |S|^2 - N^2 / Pr_t))

    with |S|^2 = 0.5 * mech_prod the square of the strain rate magnitude,
    N^2 = bruvais the Brunt-Vaisala frequency squared, and
    1 / Pr_t = rturb_prandtl the reciprocal turbulent Prandtl number.
    """
    return sqrt(maximum(wpfloat("0.0"), wpfloat("0.5") * mech_prod - rturb_prandtl * bruvais))


@gtx.field_operator
def _compute_stability_term_louis(
    mech_prod: fa.CellKHalfField[wpfloat],
    bruvais: fa.CellKHalfField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    """
    Compute the stability correction term for the eddy viscosity based on the
    stability correction function of Louis (1979):

        Ri = 2 * N^2 / max(eps, mech_prod)
        stability_function = max(1 - Ri / Pr_t,
                                 min(1, (1 / (1 + b * scaling * |Ri|))^4))
        stability_term = sqrt(0.5 * mech_prod * stability_function)
    """
    # avoids a division by zero in the Richardson number (``eps_louis`` in mo_tmx_smagorinsky.f90)
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
    mech_prod: fa.CellKHalfField[wpfloat],
    bruvais: fa.CellKHalfField[wpfloat],
    rho_ic: fa.CellKHalfField[wpfloat],
    mixing_length_sq: fa.CellKHalfField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    nlev: gtx.int32,
) -> tuple[fa.CellKHalfField[wpfloat], fa.CellKHalfField[wpfloat]]:
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
        stability_classic = _compute_stability_term_classic(
            mech_prod=mech_prod, bruvais=bruvais, rturb_prandtl=rturb_prandtl
        )
        stability_louis = _compute_stability_term_louis(
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
        stability_term = _compute_stability_term_classic(
            mech_prod=mech_prod, bruvais=bruvais, rturb_prandtl=rturb_prandtl
        )

    km = rho_ic * mixing_length_sq * stability_term
    km_ic = with_boundaries_on_half_levels_on_cells(
        top=km(dims.KHalfDim + 1), interior=km, bottom=km(dims.KHalfDim - 1), nlev=nlev
    )
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_smagorinsky_viscosity(
    mech_prod: fa.CellKHalfField[wpfloat],
    bruvais: fa.CellKHalfField[wpfloat],
    rho_ic: fa.CellKHalfField[wpfloat],
    mixing_length_sq: fa.CellKHalfField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    km_ic: fa.CellKHalfField[wpfloat],
    kh_ic: fa.CellKHalfField[wpfloat],
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
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _assign_constant_viscosity(
    rho_ic: fa.CellKHalfField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    nlev: gtx.int32,
) -> tuple[fa.CellKHalfField[wpfloat], fa.CellKHalfField[wpfloat]]:
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
        rho_ic: air density at half-level cell centers
        km_const: constant kinematic eddy viscosity
        rturb_prandtl: reciprocal turbulent Prandtl number
        nlev: number of full levels

    Returns:
        eddy viscosity km_ic and eddy diffusivity kh_ic at half levels
    """
    km = rho_ic * km_const
    km_ic = with_boundaries_on_half_levels_on_cells(
        top=km(dims.KHalfDim + 1), interior=km, bottom=km(dims.KHalfDim - 1), nlev=nlev
    )
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def assign_constant_viscosity(
    rho_ic: fa.CellKHalfField[wpfloat],
    km_ic: fa.CellKHalfField[wpfloat],
    kh_ic: fa.CellKHalfField[wpfloat],
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
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


# ---------------------------------------------------------------------------
# Compute_diagnostics: the eddy viscosity on cells, vertices and edges
# ---------------------------------------------------------------------------
@gtx.field_operator
def _interpolate_km(
    km_ic: fa.CellKHalfField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    km_min: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.VertexKHalfField[wpfloat], fa.EdgeKHalfField[wpfloat]]:
    """
    Interpolate the eddy viscosity from half-level cell centers to full-level cells,
    half-level vertices and half-level edges, with the minimum-viscosity floor ``km_min``.

    Port of ``interpolate_eddy_viscosity2cell``, ``interpolate_eddy_viscosity2half_vertex``
    and ``interpolate_eddy_viscosity2half_edge`` in ICON's ``mo_vdf_atmo.f90``. The Fortran
    applies the floor to the whole vertex and edge arrays; here it acts on the program
    domain only.
    """
    return (
        maximum(km_min, average_level_plus1_on_cells(km_ic)),
        maximum(km_min, _compute_cell_2_vertex_interpolation(km_ic, cells_aw_verts)),
        maximum(km_min, _cell_2_edge_interpolation_on_half_levels(km_ic, c_lin_e)),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km(
    km_ic: fa.CellKHalfField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    km_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKHalfField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    km_min: wpfloat,
    cell_start: gtx.int32,
    cell_end: gtx.int32,
    vertex_start: gtx.int32,
    vertex_end: gtx.int32,
    edge_start: gtx.int32,
    edge_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
    vertical_end_half: gtx.int32,
) -> None:
    _interpolate_km(
        km_ic=km_ic,
        cells_aw_verts=cells_aw_verts,
        c_lin_e=c_lin_e,
        km_min=km_min,
        out=(km_c, km_iv, km_ie),
        domain=(
            {
                dims.CellDim: (cell_start, cell_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.VertexDim: (vertex_start, vertex_end),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
            {
                dims.EdgeDim: (edge_start, edge_end),
                dims.KHalfDim: (vertical_start, vertical_end_half),
            },
        ),
    )

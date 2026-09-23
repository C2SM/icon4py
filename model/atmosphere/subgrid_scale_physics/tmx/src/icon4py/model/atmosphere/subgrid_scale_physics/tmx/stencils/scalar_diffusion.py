# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import broadcast, neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.vertical_diffusion import (
    _assemble_vertical_diffusion_matrix_on_cells,
    _solve_implicit_vertical_diffusion_on_cells,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import C2E, E2C, C2EDim
from icon4py.model.common.physics.thermodynamics.compute_energy import (
    _compute_dry_static_energy,
    compute_internal_energy_per_area,
)
from icon4py.model.common.physics.thermodynamics.compute_temperature import (
    compute_temperature_from_internal_energy_per_area,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _assemble_scalar_diffusion_matrix(
    diffusivity: fa.CellKHalfField[wpfloat],
    inv_dz: fa.CellKHalfField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    prefactor: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    return _assemble_vertical_diffusion_matrix_on_cells(
        diffusivity, inv_dz, wpfloat("1.0") / air_mass, prefactor, minlvl, maxlvl
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def assemble_scalar_diffusion_matrix(
    diffusivity: fa.CellKHalfField[wpfloat],
    inv_dz: fa.CellKHalfField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    prefactor: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _assemble_scalar_diffusion_matrix(
        diffusivity,
        inv_dz,
        air_mass,
        prefactor,
        vertical_start,
        vertical_end - 1,
        out=(a, b, c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _diffuse_scalar(
    var: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rturb_prandtl: wpfloat,
    prefactor: wpfloat,
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    New value and tendency of a cell scalar after one diffusion step: implicit in the vertical
    with matrix (a, b, c), explicit and conservative in the horizontal.

    var must be valid on the halo cells.
    """
    zero = wpfloat("0.0") * var
    vertical_tend = _solve_implicit_vertical_diffusion_on_cells(var, a, b, c, rhs, zero, dtime)
    flux = (
        wpfloat("0.5")
        * prefactor
        * rturb_prandtl
        * (km_ie(dims.KDim - 0.5) + km_ie(dims.KDim + 0.5))
        * inv_dual_edge_length
        * (var(E2C[1]) - var(E2C[0]))
    )
    tend = vertical_tend + neighbor_sum(flux(C2E) * geofac_div, axis=C2EDim) / rho
    return var + tend * dtime, tend


@gtx.field_operator
def _diffuse_tracer(
    var: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    surface_flux: fa.CellField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rturb_prandtl: wpfloat,
    prefactor: wpfloat,
    dtime: wpfloat,
    maxlvl: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """:func:`_diffuse_scalar` with the surface flux entering the bottom row maxlvl."""
    rhs = concat_where(
        dims.KDim < maxlvl,
        wpfloat("0.0") * var,
        wpfloat("0.0") - surface_flux * prefactor * (wpfloat("1.0") / air_mass),
    )
    return _diffuse_scalar(
        var,
        a,
        b,
        c,
        rhs,
        rho,
        km_ie,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        prefactor,
        dtime,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diffuse_tracer(
    var: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    surface_flux: fa.CellField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    new_var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    rturb_prandtl: wpfloat,
    prefactor: wpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _diffuse_tracer(
        var,
        a,
        b,
        c,
        surface_flux,
        air_mass,
        rho,
        km_ie,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        prefactor,
        dtime,
        vertical_end - 1,
        out=(new_var, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_energy_from_temperature(
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
    use_internal_energy: bool,
) -> fa.CellKField[wpfloat]:
    """
    Specific energy diffused by the heat diffusion: the internal energy plus cvd / cpd times the
    geopotential above ground, or the dry static energy.
    """
    if use_internal_energy:
        one = broadcast(wpfloat("1.0"), (dims.CellDim, dims.KDim))
        energy = (
            compute_internal_energy_per_area(temperature, qv, qc + qr, qi + qs + qg, one, one)
            + grav * height_above_ground * PhysicsConstants.cvd / PhysicsConstants.cpd
        )
    else:
        energy = _compute_dry_static_energy(temperature, height_above_ground, grav)
    return energy


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_energy_from_temperature(
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    energy: fa.CellKField[wpfloat],
    grav: wpfloat,
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_energy_from_temperature(
        temperature,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        height_above_ground,
        grav,
        use_internal_energy,
        out=energy,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _diffuse_energy_and_update_temperature(
    energy: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rturb_prandtl: wpfloat,
    prefactor: wpfloat,
    grav: wpfloat,
    dtime: wpfloat,
    maxlvl: gtx.int32,
    use_internal_energy: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    New temperature and its tendency after one diffusion step of the energy of
    :func:`_compute_energy_from_temperature`, with the surface energy flux entering the bottom
    row maxlvl.

    The new temperature is recovered with the new qv, qc and qi.
    """
    inv_air_mass = wpfloat("1.0") / air_mass
    # only the bottom row of the surface term is used, where 'temperature' is that of the
    # lowest level
    if use_internal_energy:
        surface_term = (
            wpfloat("0.0")
            - (
                sensible_heat_flux
                + temperature * evapotranspiration * (PhysicsConstants.cvv - PhysicsConstants.cvd)
            )
            * prefactor
            * inv_air_mass
        )
    else:
        surface_term = (
            wpfloat("0.0")
            - sensible_heat_flux
            * PhysicsConstants.cpd
            / PhysicsConstants.cvd
            * prefactor
            * inv_air_mass
        )
    rhs = concat_where(dims.KDim < maxlvl, wpfloat("0.0") * energy, surface_term)
    new_energy, _ = _diffuse_scalar(
        energy,
        a,
        b,
        c,
        rhs,
        rho,
        km_ie,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        prefactor,
        dtime,
    )
    if use_internal_energy:
        one = broadcast(wpfloat("1.0"), (dims.CellDim, dims.KDim))
        new_temperature = compute_temperature_from_internal_energy_per_area(
            new_energy - grav * height_above_ground * PhysicsConstants.cvd / PhysicsConstants.cpd,
            new_qv,
            new_qc + qr,
            new_qi + qs + qg,
            one,
            one,
        )
    else:
        new_temperature = (new_energy - grav * height_above_ground) / PhysicsConstants.cpd
    return new_temperature, (new_temperature - temperature) * (wpfloat("1.0") / dtime)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diffuse_energy_and_update_temperature(
    energy: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    rturb_prandtl: wpfloat,
    prefactor: wpfloat,
    grav: wpfloat,
    dtime: wpfloat,
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _diffuse_energy_and_update_temperature(
        energy,
        a,
        b,
        c,
        sensible_heat_flux,
        evapotranspiration,
        temperature,
        new_qv,
        new_qc,
        new_qi,
        qr,
        qs,
        qg,
        air_mass,
        rho,
        km_ie,
        height_above_ground,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        prefactor,
        grav,
        dtime,
        vertical_end - 1,
        use_internal_energy,
        out=(new_temperature, tend_temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

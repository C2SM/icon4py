# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stencils of the tmx scalar (hydrometeor and heat) diffusion.

Ports ``Compute_diffusion_hydrometeors`` (mo_vdf.f90 l. 585, run by
:meth:`Tmx.run_hydrometeor_diffusion`) and ``Compute_diffusion_temperature``
(mo_vdf.f90 l. 912, run by :meth:`Tmx.run_temperature_diffusion`). Both diffuse
a cell scalar: implicit (or explicit) vertical diffusion with the surface flux
entering through the bottom-row right-hand side, followed by conservative
horizontal nabla2 diffusion and the state update.

The right-hand side and the vertical solve are fused into one program (see
:mod:`vertical_diffusion`), and the matrix rows into another; of the Fortran
scratch arrays only ``inv_mair`` and the matrix rows survive as fields, because
both are loop-invariant and shared by the three hydrometeors.
"""

import gt4py.next as gtx
from gt4py.next import broadcast, neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.vertical_diffusion import (
    _apply_explicit_vertical_diffusion_cells,
    _prepare_tridiagonal_matrix_cells,
    _solve_vertical_diffusion_cells,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import C2E, E2C, C2EDim, KDim
from icon4py.model.common.math.operators import _compute_reciprocal_on_cell_k
from icon4py.model.common.physics.thermodynamics import _compute_temperature_from_internal_energy
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_flux_rhs(
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    prefac: wpfloat,
    maxlvl: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """
    Right-hand side of the scalar vertical diffusion solve.

    Port of the right-hand-side rows of 'Compute_diffusion_hydrometeors' and
    'Compute_diffusion_temperature' (mo_vdf.f90):

        rhs(maxlvl) = - sfc_flx * prefac * inv_mair(maxlvl)

    with ``prefac = 1`` for the hydrometeors and ``prefac = zfactor``
    (``scale_turb_energy_flux`` if enabled, else 1) for the energy. All other
    rows are zero: the Fortran zero-initializes ``rhs`` and only writes the
    bottom row and the top row ``rhs(1) = + top_flx * inv_mair(1)``, where
    ``top_flx`` is always zero in tmx.

    Args:
        sfc_flx: grid-mean surface flux of the diffused quantity (2D cell field)
        inv_air_mass: inverse air mass per unit area at full levels [m^2/kg]
        prefac: scaling factor of the turbulent flux
        maxlvl: bottom row of the solve (``nlev - 1``)

    Returns:
        right-hand side of the vertical diffusion solve at all full levels
    """
    bottom = wpfloat("0.0") - sfc_flx * prefac * inv_air_mass
    return concat_where(dims.KDim < maxlvl, inv_air_mass * wpfloat("0.0"), bottom)


@gtx.field_operator
def _prepare_scalar_diffusion_matrix(
    air_mass: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    zprefac: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Inverse air mass and tridiagonal matrix of the scalar vertical diffusion.

    Fuses the ``inv_mair`` loops of mo_vdf.f90 with
    ``prepare_diffusion_matrix_wp`` (mo_vdf_atmo.f90), which scales the matrix
    rows with the inverse air mass. ``inv_air_mass`` is also needed by the
    surface-flux right-hand side of the solve and is therefore returned.
    """
    inv_air_mass = _compute_reciprocal_on_cell_k(input_field=air_mass)
    a, b, c = _prepare_tridiagonal_matrix_cells(
        inv_mair=inv_air_mass,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        minlvl=minlvl,
        maxlvl=maxlvl,
    )
    return inv_air_mass, a, b, c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_scalar_diffusion_matrix(
    air_mass: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    zprefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _prepare_scalar_diffusion_matrix(
        air_mass=air_mass,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        minlvl=vertical_start,
        maxlvl=vertical_end - 1,
        out=(inv_air_mass, a, b, c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _solve_scalar_vertical_diffusion(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    prefac: wpfloat,
    dtime: wpfloat,
    maxlvl: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """Surface-flux right-hand side and implicit vertical solve of a cell scalar."""
    rhs = _compute_surface_flux_rhs(
        sfc_flx=sfc_flx, inv_air_mass=inv_air_mass, prefac=prefac, maxlvl=maxlvl
    )
    return _solve_vertical_diffusion_cells(a=a, b=b, c=c, rhs=rhs, var=var, tend=tend, dtime=dtime)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_scalar_vertical_diffusion(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    prefac: wpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_scalar_vertical_diffusion(
        a=a,
        b=b,
        c=c,
        sfc_flx=sfc_flx,
        inv_air_mass=inv_air_mass,
        var=var,
        tend=tend,
        prefac=prefac,
        dtime=dtime,
        maxlvl=vertical_end - 1,
        out=tend,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _apply_explicit_scalar_vertical_diffusion(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    prefac: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """Surface-flux right-hand side and explicit vertical diffusion of a cell scalar."""
    rhs = _compute_surface_flux_rhs(
        sfc_flx=sfc_flx, inv_air_mass=inv_air_mass, prefac=prefac, maxlvl=maxlvl
    )
    return _apply_explicit_vertical_diffusion_cells(
        a=a, b=b, c=c, rhs=rhs, var=var, tend=tend, minlvl=minlvl, maxlvl=maxlvl
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_explicit_scalar_vertical_diffusion(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    prefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_explicit_scalar_vertical_diffusion(
        a=a,
        b=b,
        c=c,
        sfc_flx=sfc_flx,
        inv_air_mass=inv_air_mass,
        var=var,
        tend=tend,
        prefac=prefac,
        minlvl=vertical_start,
        maxlvl=vertical_end - 1,
        out=tend,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_scalar_nabla2_flux(
    scalar: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    rturb_prandtl: wpfloat,
    prefac: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """
    Compute the horizontal turbulent diffusion flux of a cell scalar at edges.

    Port of the ``nabla2_e`` loops ("compute kh_ie * grad_horiz(state)") of the
    conservative horizontal diffusion in 'Compute_diffusion_hydrometeors' and
    'Compute_diffusion_temperature' (mo_vdf.f90):

        nabla2_e(k) = 0.5 * prefac * rturb_prandtl * (km_ie(k) + km_ie(k+1))
                      * inv_dual_edge_length
                      * (scalar(E2C[1]) - scalar(E2C[0]))

    ``0.5 * rturb_prandtl * (km_ie(k) + km_ie(k+1))`` is the turbulent
    diffusivity ``kh`` averaged from the adjacent half levels to the full level
    ``k``, and ``inv_dual_edge_length * (scalar(E2C[1]) - scalar(E2C[0]))`` is
    the horizontal gradient normal to the edge. ``prefac = 1`` for the
    hydrometeors and ``prefac = zfactor`` (``scale_turb_energy_flux`` if
    enabled, else 1) for the energy.

    Args:
        scalar: diffused cell scalar at full levels (halo values must be valid)
        km_ie: turbulent viscosity at half-level edges [m^2/s] (nlev + 1 rows)
        inv_dual_edge_length: inverse dual edge length [1/m]
        rturb_prandtl: reciprocal turbulent Prandtl number
        prefac: scaling factor of the turbulent flux

    Returns:
        horizontal turbulent diffusion flux at full-level edges
    """
    return (
        wpfloat("0.5")
        * prefac
        * rturb_prandtl
        * (km_ie + km_ie(KDim + 1))
        * inv_dual_edge_length
        * (scalar(E2C[1]) - scalar(E2C[0]))
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_scalar_nabla2_flux(
    scalar: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    rturb_prandtl: wpfloat,
    prefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_scalar_nabla2_flux(
        scalar=scalar,
        km_ie=km_ie,
        inv_dual_edge_length=inv_dual_edge_length,
        rturb_prandtl=rturb_prandtl,
        prefac=prefac,
        out=nabla2_flux,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _apply_horizontal_diffusion_and_update_scalar(
    scalar: fa.CellKField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rho: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Apply the horizontal turbulent diffusion tendency and update a cell scalar.

    Port of the flux divergence and update loops of the conservative horizontal
    diffusion in 'Compute_diffusion_hydrometeors' and
    'Compute_diffusion_temperature' (mo_vdf.f90):

        hori_tend = (sum_{e in C2E} nabla2_flux(e) * geofac_div(e)) / rho
        tend      = tend + hori_tend
        new       = scalar + tend * dtime

    ``tend`` holds the vertical diffusion tendency on entry (written by the
    vertical diffusion solver) and the total (vertical + horizontal) tendency
    on exit.

    Args:
        scalar: diffused cell scalar at full levels (old state)
        nabla2_flux: horizontal turbulent diffusion flux at full-level edges
        geofac_div: geometric factors of the cell-centered edge-flux divergence
        rho: air density at full levels [kg/m^3]
        tend: vertical diffusion tendency of the scalar at full levels
        dtime: time step [s]

    Returns:
        (updated scalar, total diffusion tendency) at full levels
    """
    hori_tend = neighbor_sum(nabla2_flux(C2E) * geofac_div, axis=C2EDim) / rho
    new_tend = tend + hori_tend
    new_scalar = scalar + new_tend * dtime
    return new_scalar, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_horizontal_diffusion_and_update_scalar(
    scalar: fa.CellKField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rho: fa.CellKField[wpfloat],
    new_scalar: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_horizontal_diffusion_and_update_scalar(
        scalar=scalar,
        nabla2_flux=nabla2_flux,
        geofac_div=geofac_div,
        rho=rho,
        tend=tend,
        dtime=dtime,
        out=(new_scalar, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_temperature_from_energy_and_tendency(
    energy: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    use_internal_energy: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the new temperature from the diffused energy and the resulting
    temperature tendency.

    Port of 'energy_to_temp' (mo_vdf_atmo.f90) and the final tendency loop of
    'Compute_diffusion_temperature' (mo_vdf.f90):

    - dry static energy (``energy_type = 1``, ``use_internal_energy = False``,
      'compute_temp_from_static_energy'):

          new_temperature = (energy - grav * height_above_ground) / cpd

    - internal energy + geopotential above ground (``energy_type = 2``,
      ``use_internal_energy = True``,
      'compute_temperature_from_internal_energy'):

          u               = energy - grav * height_above_ground * cvd / cpd
          new_temperature = T_from_internal_energy(u, qv, qc + qr,
                                                   qi + qs + qg,
                                                   rho = 1, dz = 1)

      with ``T_from_internal_energy`` from mo_aes_thermo.f90 (ported in
      :mod:`icon4py.model.common.physics.thermodynamics`). The Fortran call
      site uses the *new* moisture state (``use_new_moisture_state=.TRUE.``,
      the tracers updated by the hydrometeor diffusion); qr, qs and qg are not
      diffused and have no new state.

    In both cases the temperature tendency is

          tend_temperature = (new_temperature - temperature) / dtime

    ``use_internal_energy`` is a scalar configuration flag; it can be passed as
    a static (compile-time) argument so that only the selected variant is
    compiled.

    Args:
        energy: diffused (new) energy at full levels [J/kg]
        temperature: air temperature before the diffusion at full levels [K]
        qv: new specific humidity [kg/kg]
        qc: new cloud water mixing ratio [kg/kg]
        qi: new cloud ice mixing ratio [kg/kg]
        qr: rain mixing ratio [kg/kg]
        qs: snow mixing ratio [kg/kg]
        qg: graupel mixing ratio [kg/kg]
        height_above_ground: height of the full levels above the surface [m]
        grav: gravitational acceleration [m/s^2]
        dtime: time step [s]
        use_internal_energy: True for internal energy, False for dry static energy

    Returns:
        (new temperature, temperature tendency) at full levels
    """
    if use_internal_energy:
        one = broadcast(wpfloat("1.0"), (dims.CellDim, dims.KDim))
        q_liquid = qc + qr
        q_solid = qi + qs + qg
        u = energy - grav * height_above_ground * PhysicsConstants.cvd / PhysicsConstants.cpd
        new_temperature = _compute_temperature_from_internal_energy(
            u=u, qv=qv, qliq=q_liquid, qice=q_solid, rho=one, dz=one
        )
    else:
        new_temperature = (energy - grav * height_above_ground) / PhysicsConstants.cpd
    rdtime = wpfloat("1.0") / dtime
    tend_temperature = (new_temperature - temperature) * rdtime
    return new_temperature, tend_temperature


@gtx.field_operator
def _apply_horizontal_diffusion_and_update_temperature(
    energy: fa.CellKField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rho: fa.CellKField[wpfloat],
    tend_energy: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    use_internal_energy: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Horizontal energy diffusion and recovery of the new temperature.

    Fuses the flux divergence and update loops of the conservative horizontal
    diffusion with 'energy_to_temp' and the final tendency loop of
    'Compute_diffusion_temperature' (mo_vdf.f90): the new energy only enters
    the temperature and is not materialized.

    Returns:
        total energy diffusion tendency, new temperature and temperature
        tendency
    """
    new_energy, new_tend_energy = _apply_horizontal_diffusion_and_update_scalar(
        scalar=energy,
        nabla2_flux=nabla2_flux,
        geofac_div=geofac_div,
        rho=rho,
        tend=tend_energy,
        dtime=dtime,
    )
    new_temperature, tend_temperature = _compute_temperature_from_energy_and_tendency(
        energy=new_energy,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        height_above_ground=height_above_ground,
        grav=grav,
        dtime=dtime,
        use_internal_energy=use_internal_energy,
    )
    return new_tend_energy, new_temperature, tend_temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_horizontal_diffusion_and_update_temperature(
    energy: fa.CellKField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rho: fa.CellKField[wpfloat],
    tend_energy: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_horizontal_diffusion_and_update_temperature(
        energy=energy,
        nabla2_flux=nabla2_flux,
        geofac_div=geofac_div,
        rho=rho,
        tend_energy=tend_energy,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        height_above_ground=height_above_ground,
        grav=grav,
        dtime=dtime,
        use_internal_energy=use_internal_energy,
        out=(tend_energy, new_temperature, tend_temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_surface_energy_flux(
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    use_internal_energy: bool,
) -> fa.CellField[wpfloat]:
    """
    Compute the grid-mean surface flux of the energy diffused by the tmx heat
    diffusion.

    Port of 'compute_flux_x' (mo_vdf_atmo.f90) with the ``ufts`` / ``ufvs``
    energy fluxes inlined from 'compute_energy_fluxes' (mo_tmx_surface.f90,
    called on the grid-mean fluxes at the end of the surface Compute in
    mo_vdf_sfc.f90):

    - dry static energy (``energy_type = 1``, ``use_internal_energy = False``):

          flux_x = sensible_heat_flux * cpd / cvd

    - internal energy (``energy_type = 2``, ``use_internal_energy = True``):

          ufts   = sensible_heat_flux
          ufvs   = temperature_sfc * evapotranspiration * (cvv - cvd)
          flux_x = ufts + ufvs

      ``ufts`` is the surface energy flux from thermal exchange and ``ufvs``
      the one from vapor exchange.

    ``use_internal_energy`` is a scalar configuration flag; it can be passed as
    a static (compile-time) argument so that only the selected variant is
    compiled.

    Args:
        sensible_heat_flux: grid-mean surface sensible heat flux (``shfl``) [W/m^2]
        evapotranspiration: grid-mean surface evapotranspiration flux [kg/(m^2 s)]
        temperature_sfc: air temperature at the lowest full level (``ta(nlev)``) [K]
        use_internal_energy: True for internal energy, False for dry static energy

    Returns:
        grid-mean surface energy flux (``flux_x``) [W/m^2]
    """
    if use_internal_energy:
        ufts = sensible_heat_flux
        ufvs = temperature_sfc * evapotranspiration * (PhysicsConstants.cvv - PhysicsConstants.cvd)
        flux_x = ufts + ufvs
    else:
        flux_x = sensible_heat_flux * PhysicsConstants.cpd / PhysicsConstants.cvd
    return flux_x


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_energy_flux(
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    flux_x: fa.CellField[wpfloat],
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_energy_flux(
        sensible_heat_flux=sensible_heat_flux,
        evapotranspiration=evapotranspiration,
        temperature_sfc=temperature_sfc,
        use_internal_energy=use_internal_energy,
        out=flux_x,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Everything 'vertdiff' builds once: the parts of the matrix no variable and no type changes.

Translated from 'icon/src/atm_phy_schemes/turb_vertdiff.f90', SUBROUTINE 'vertdiff' (:536-542
and :614-634 at icon commit 26d6b98cce), and from 'turb_utilities.f90', SUBROUTINE
'vert_grad_diff' (:2438-2455), which 'vertdiff' reaches at :747-763 and which runs the block
under 'linisetup' -- for the first variable of the first type only. The scientific commentary in
both files is by Matthias Raschendorfer (DWD).

ONE PROGRAM, SIX STATEMENTS, TWO DOMAINS. In source order,

  1. the air density at the lower boundary of the Prandtl layer, 'rhon(:,ke1)';
  2. the Exner factor there, 'eprs(:,ke1)' -- the stage's ONLY transcendental;
  3. the discretisation momentum 'disc_mom', main levels 1..ke;
  4. the diffusion depth 'diff_dep', interior flux levels 2..ke;
  5. the effective surface temperature gradient 'zvari(:,ke1,tem)';
  6. the effective surface vapour gradient 'zvari(:,ke1,vap)'.

WHY THESE SIX AND NOT SOME OTHER SIX. They are exactly the quantities 'vertdiff' computes before
it enters its loop over variable types, so they are the ones a merged unit can hold without
being run more often than ICON runs them. Everything after them depends on 'tkv'/'tsv' and is
therefore per type ('prep_impl_vert_diff') or per variable ('calc_impl_vert_diff').

STATEMENTS 5 AND 6 READ WHAT STATEMENTS 1 AND 2 WROTE, and that is the one real edge the source
order now carries: the surface gradients divide by 'rhon(:,ke1)*tkvh(:,ke1)', and the
temperature one additionally by 'c_pd*eprs(:,ke1)'. While these were two programs the ordering
lived in '_prepare_the_diffusion_matrix'; it is statement order now. Both reads are POINTWISE
reads of a parameter an earlier statement wrote, which is the shape GT4Py orders correctly on
every backend -- see 'solve_turb_budgets' for the shape that it does not.

THE TWO ROUNDINGS OF THE DISCRETISATION MOMENTUM ARE NOT NEGOTIABLE, and neither is the grouping
of the diffusion depth or of the two gradient denominators; each is noted on its operator. They
are the reason this stage is bit-exact against ICON in seventeen of its eighteen programs.
"""

import enum

import gt4py.next as gtx
from gt4py.next import broadcast, exp, log

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


class SurfaceFluxConstants(ta.wpfloat, enum.Enum):
    """The one physical constant this stencil reads.

    An enum because GT4Py can only fold a closure variable that is an attribute of an enum class
    or of an 'eve.FrozenNamespace'; see the docstring of
    'thermodynamic_functions.ThermoConstants', which holds every other constant the turbulence
    stencils use. 'c_pd' is not among them, since nothing before 'vertdiff' needed it.
    """

    #: 'cp_d', the specific heat of dry air at constant pressure [J/K/kg].
    CP_D = constants.CPD


@gtx.field_operator
def _compute_surface_air_density(
    surface_pressure: fa.CellField[wpfloat],
    surface_specific_humidity: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'rhon(:,ke1)': the ideal-gas density of moist air at the ground [kg/m3].

    'rhon' is 'INTENT(INOUT)' and this is the only place 'vertdiff' writes it; every other row
    is what 'turbdiff' left. The Fortran's own note is worth carrying: in the turbulence model
    'rhon(:,ke1)' belongs to the lower boundary of the Prandtl layer rather than to the surface
    level, but 'vert_grad_diff' uses it as a surface-level value.

    The virtual factor is written and multiplied exactly as the Fortran does -- 'r_d*virt*t_g'
    associates to the left, and the quotient is formed once -- because any regrouping of the
    denominator is a rounding this port would not be able to explain away.
    """
    virtual_factor = wpfloat("1.0") + ThermoConstants.RVD_M_O * surface_specific_humidity
    return broadcast(
        surface_pressure / (ThermoConstants.RD * virtual_factor * surface_temperature),
        (dims.CellDim, dims.KDim),
    )


@gtx.field_operator
def _compute_surface_exner_factor(
    surface_pressure: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'eprs(:,ke1) = zexner(ps)': the Exner factor '(p_s/p0)**(R_d/c_pd)' [-].

    'EXP(rdocp*LOG(...))' rather than '**', as 'thermodynamic_functions._exner_factor' explains
    -- and as the Fortran itself writes it. This is the one transcendental in 'vertdiff'.
    """
    return broadcast(
        exp(ThermoConstants.RDOCP * log(surface_pressure / ThermoConstants.P0REF)),
        (dims.CellDim, dims.KDim),
    )


@gtx.field_operator
def _compute_discretisation_momentum(
    air_density_at_main_levels: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'disc_mom(k) = rho(k)*(hhl(k) - hhl(k+1))*fr_var' [kg/m2/s].

    The mass per unit area of a model layer divided by the time step: the diagonal of the
    tridiagonal system before any diffusion momentum is added to it. It is a function of the
    grid and of the density alone, which is why 'vert_grad_diff' computes it once, under
    'linisetup', and every later variable and type reuses it.

    TWO ROUNDINGS THAT ARE NOT NEGOTIABLE. 'fr_var' is '1/dt_var', formed once by the caller and
    MULTIPLIED by; 'rho*dz/dt_var' is a different number. And the product associates to the
    left, '(rho*dz)*fr_var'. Both were measured: getting either wrong puts 'disc_mom' one ulp
    off on 250707 of 662080 values, and the error survives into every tendency the scheme
    produces.

    The layer depth is not a field. The Fortran parks it in the 'expl_mom' storage, which is why
    'expl_mom(:,1)' still holds the top layer's depth at the exit savepoint -- the one row the
    diffusion momentum never overwrites. That is an artefact of storage reuse and this port does
    not reproduce it. The density here is 'rhoh', the MAIN-level density, and not the 'rhon'
    this program writes a surface row of.
    """
    layer_depth = half_level_height - half_level_height(Koff[1])
    return air_density_at_main_levels * layer_depth * reciprocal_time_step


@gtx.field_operator
def _compute_diffusion_depth(half_level_height: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    """'0.5*(dz(k-1) + dz(k))' [m], with 'dz(k) = hhl(k) - hhl(k+1)'.

    The two depths are formed as differences of half-level heights and added before the halving,
    which is the Fortran's grouping; '0.5*dz(k-1) + 0.5*dz(k)' would round twice.
    """
    depth_above = half_level_height(Koff[-1]) - half_level_height
    depth_here = half_level_height - half_level_height(Koff[1])
    return wpfloat("0.5") * (depth_above + depth_here)


@gtx.field_operator
def _compute_surface_temperature_gradient(
    sensible_heat_flux: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_coefficient: fa.CellKField[wpfloat],
    surface_exner_factor: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'shfl_s/(rhon*tkvh*c_pd*eprs)' [K/m], with the denominator associated to the left."""
    transfer_momentum = air_density * diffusion_coefficient
    return broadcast(sensible_heat_flux, (dims.CellDim, dims.KDim)) / (
        transfer_momentum * SurfaceFluxConstants.CP_D * surface_exner_factor
    )


@gtx.field_operator
def _compute_surface_vapour_gradient(
    water_vapour_flux: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_coefficient: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'qvfl_s/(rhon*tkvh)' [1/m]."""
    transfer_momentum = air_density * diffusion_coefficient
    return broadcast(water_vapour_flux, (dims.CellDim, dims.KDim)) / transfer_momentum


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_the_vertical_diffusion_matrix(
    surface_pressure: fa.CellField[wpfloat],
    surface_specific_humidity: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
    air_density_at_main_levels: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
    diffusion_coefficient: fa.CellKField[wpfloat],
    sensible_heat_flux: fa.CellField[wpfloat],
    water_vapour_flux: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    surface_exner_factor: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    surface_temperature_gradient: fa.CellKField[wpfloat],
    surface_vapour_gradient: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Build the part of the diffusion matrix that neither variable type nor variable changes.

    THE TWO VERTICAL RANGES, both arithmetic on the one pair the granule binds.
    '(vertical_start, vertical_end)' is the main-level range of the discretisation momentum: the
    Fortran 'disc_mom(:,k_hi)' statement followed by 'DO k = k_hi+1, k_lw' with 'k_hi = 1' and
    'k_lw = ke', which is 'vertical_start = 0, vertical_end = nlev'. On it,

      * the diffusion depth runs from 'vertical_start + 1', because row 0 has no level above it
        and is not a flux level. Its surface row is NOT written here: 'diff_dep(:,k_sf)' is
        'tkv/tsv', a per-variable-type transfer depth, and belongs to
        'compute_surface_diffusion_momentum_and_depth';
      * the four surface-row statements run on '(vertical_end, vertical_end + 1)', which is the
        Fortran 'ke1' -- one row, four quantities.

    'air_density' is the one field of the granule that both stages write, and this program is
    where 'vertdiff' does it: 'rhon(:,ke1)' becomes the ideal-gas density of the ground, which is
    a different quantity from the Prandtl-layer boundary value 'turbdiff' left there. Every other
    row is untouched.

    The two surface gradients land in 'zvari(:,ke1,tem)' and 'zvari(:,ke1,vap)', the granule's
    gradient fields, because that is the storage the Fortran uses and because their consumer
    reads them from there. They run for a variable whose 'lsfli(n)' is true, which under
    'lsfluse' -- the NWP interface passes it '.TRUE.' -- is temperature and water vapour and
    nothing else, so the two outputs are the complete set. The solve later overwrites both rows
    with the explicit surface flux.

    Args:
        surface_pressure: 'ps' [Pa].
        surface_specific_humidity: 'qv_s' [kg/kg].
        surface_temperature: 't_g', the weighted surface temperature [K].
        air_density_at_main_levels: 'rho' ('rhoh') [kg/m3], main levels.
        half_level_height: 'hhl' [m] ('p_metrics%z_ifc'); read one level up and one level down.
        reciprocal_time_step: 'fr_var = 1/dt_var' [1/s]. A scalar, and formed by the caller so
            that the division happens exactly once for the whole scheme.
        diffusion_coefficient: 'tkvh' [m2/s], the SCALAR type's; only the surface row is read.
        sensible_heat_flux: 'shfl_s' [W/m2], positive downward.
        water_vapour_flux: 'qvfl_s' [kg/m2/s], positive downward.
        air_density: In-out, 'rhon' on half levels [kg/m3]; only the surface row is written, and
            it is read back by the two gradient statements.
        surface_exner_factor: Output, 'eprs' [-]; surface row only.
        discretisation_momentum: Output, 'disc_mom' [kg/m2/s], main levels.
        diffusion_depth: Output, 'diff_dep' [m], interior flux levels.
        surface_temperature_gradient: In-out, 'zvari(:,:,tem)' [K/m]; surface row only.
        surface_vapour_gradient: In-out, 'zvari(:,:,vap)' [1/m]; surface row only.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost main level; 0, mirroring Fortran 'k_hi = 1'.
        vertical_end: End of the main levels; 'nlev', mirroring Fortran 'k_lw = ke'. The surface
            row is one past it.
    """
    _compute_surface_air_density(
        surface_pressure=surface_pressure,
        surface_specific_humidity=surface_specific_humidity,
        surface_temperature=surface_temperature,
        out=air_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )
    _compute_surface_exner_factor(
        surface_pressure=surface_pressure,
        out=surface_exner_factor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )
    _compute_discretisation_momentum(
        air_density_at_main_levels=air_density_at_main_levels,
        half_level_height=half_level_height,
        reciprocal_time_step=reciprocal_time_step,
        out=discretisation_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_diffusion_depth(
        half_level_height=half_level_height,
        out=diffusion_depth,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end),
        },
    )
    _compute_surface_temperature_gradient(
        sensible_heat_flux=sensible_heat_flux,
        air_density=air_density,
        diffusion_coefficient=diffusion_coefficient,
        surface_exner_factor=surface_exner_factor,
        out=surface_temperature_gradient,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )
    _compute_surface_vapour_gradient(
        water_vapour_flux=water_vapour_flux,
        air_density=air_density,
        diffusion_coefficient=diffusion_coefficient,
        out=surface_vapour_gradient,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Section 1a) of 'turbdiff': the vertical gradients the rest of the scheme is built on.

Translated from ICON's turb_diffusion.f90, SUBROUTINE 'turbdiff', section
"1a) Berechnung der benoetigten vertikalen Gradienten und Abspeichern auf 'zvari'"
("Calculation of the required vertical gradients, and storing them in 'zvari'"), lines
1151-1203 at icon commit 26d6b98cce. The scientific commentary in that file is by Matthias
Raschendorfer (DWD), who splits the section into two blocks: "Am unteren Modellrand" ("At the
lower boundary of the model", :1151-1170) and "An den darueberliegenden Nebenflaechen" ("At the
half levels above it", :1172-1203).

ONE PROGRAM, THREE STATEMENTS, THREE DOMAINS. A '@gtx.program' body is a sequence of
field-operator calls, each with its own 'out=' and its own 'domain=', so the section's three
former programs are three statements here, each keeping the exact vertical range it had:

  1. the two surface transfer ratios 'lays', which have no vertical axis at all;
  2. the reciprocal layer depth 'hlp' and the TKE discretisation momentum 'dicke', rows 1..ke-1;
  3. the five gradients 'zvari', rows 1..ke, the last of them selected by 'concat_where'.

Statements 2 and 3 also fix the section's internal data flow in the source: the gradients divide
by 'hlp' above the surface and by 'lays' on it, and both are written by the statements above
them. Before the merge that order was a paragraph in 'run_turbdiff' asking the reader to trust
it.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_transfer_ratios(
    tvm: fa.CellField[wpfloat],
    tvh: fa.CellField[wpfloat],
    tkvm_at_surface: fa.CellField[wpfloat],
    tkvh_at_surface: fa.CellField[wpfloat],
    tfm: fa.CellField[wpfloat],
    tfh: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Compute the two ratios that turn a Prandtl-layer difference into a surface gradient.

    From the block Raschendorfer heads "Am unteren Modellrand" ("At the lower boundary of the
    model"), turb_diffusion.f90:1151-1160:

        lays(i,mom) = tvm(i) / (tkvm(i,ke1) * tfm(i))
        lays(i,sca) = tvh(i) / (tkvh(i,ke1) * tfh(i))

    NOT A TRANSLATED COMMENT -- the Fortran states the formula and not its reasoning, and the
    following is a reconstruction of why the two factors are where they are. It is offered for
    scientific review, not asserted.

    The transfer velocity is the reciprocal of the TOTAL transfer-layer resistance, from the
    surface up to the lowest main level, while 'tf' is the share of that resistance carried by
    the Prandtl layer alone ('tfm = dz_0a_m / dz_sa_m', turb_transfer.f90:1323). So 'tv / tf'
    is the reciprocal resistance of the Prandtl layer alone, and dividing by the diffusion
    coefficient turns it into a reciprocal length -- the effective depth of the Prandtl layer.

    That is the length over which the difference this ratio multiplies is taken. Section 0) put
    into the surface row of each variable its value at the LOWER EDGE OF THE PRANDTL LAYER and
    not at the surface -- for the wind components literally 'u(ke) * (1 - tfm)'
    (turb_diffusion.f90:1057-1063) -- so the difference section 1a) forms spans the Prandtl
    layer and not the whole transfer layer, and the 'tf' in this denominator is what matches
    it. The product is the gradient at the lowest half level, and multiplying that by the
    diffusion coefficient returns exactly the surface flux 'tv * (value at ke - surface value)'
    the transfer law gives.

    One ratio per variable type, and only two of them, because the Prandtl-layer resistance
    differs between momentum and scalars but not among the scalars: the Fortran indexes
    'lays' by 'ivtp(n)', which maps the two wind components to 'mom' and the three scalars
    ('tet_l', 'h2o_g', 'liq') to 'sca'. That index array is Fortran bookkeeping and is
    replaced here by two named outputs whose consumer picks the right one.

    Args:
        tvm: turbulent transfer velocity for momentum at the surface [m/s]
        tvh: turbulent transfer velocity for heat and moisture at the surface [m/s]
        tkvm_at_surface: turbulent diffusion coefficient for momentum at the lowest half
            level 'ke1' [m2/s]
        tkvh_at_surface: turbulent diffusion coefficient for scalars at the lowest half
            level 'ke1' [m2/s]
        tfm: Prandtl-layer fraction of the total transfer-layer resistance for momentum [1]
        tfh: Prandtl-layer fraction of the total transfer-layer resistance for scalars [1]

    Returns:
        surface transfer ratio for momentum [1/m], surface transfer ratio for scalars [1/m]
    """
    surface_transfer_ratio_for_momentum = tvm / (tkvm_at_surface * tfm)
    surface_transfer_ratio_for_scalars = tvh / (tkvh_at_surface * tfh)
    return surface_transfer_ratio_for_momentum, surface_transfer_ratio_for_scalars


@gtx.field_operator
def _compute_inverse_layer_depth_and_tke_discretisation_momentum(
    hhl: fa.CellKField[wpfloat],
    rhon: fa.CellKField[wpfloat],
    inverse_tke_time_step: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the reciprocal half-level layer depth and the TKE discretisation momentum.

    From the block Raschendorfer heads "An den darueberliegenden Nebenflaechen" ("At the half
    levels above it", the ones above the lower boundary treated just before),
    turb_diffusion.f90:1172-1184:

        wert       = (hhl(i,k-1) - hhl(i,k+1)) * z1d2
        hlp(i,k)   = z1 / wert
        dicke(i,k) = rhon(i,k) * wert * fr_tke

    'wert' is the depth of the layer centred on half level k, i.e. the distance between the
    two main levels that surround it, taken as half the distance between the enclosing half
    levels. It carries both quantities the rest of the scheme needs at a half level:

      * its reciprocal is the denominator of every centred vertical difference, and the next
        statement of this program uses it immediately for the gradients of the quasi-conserved
        variables;
      * multiplied by the half-level density and divided by the TKE time step it is the
        discretisation momentum of the vertical TKE diffusion, 'rho_n * dz / dt_tke'
        [kg/m2/s] -- the mass per unit area of the layer, per unit time -- which section 9)
        hands to the semi-implicit solver as 'disc_mom'.

    The Fortran computes both from one 'wert' and so does this operator, so the two outputs
    round identically; the division and the two multiplications are in the Fortran's order.

    Neither output is defined at the model top (k = 0) or at the surface (k = nlev): the
    'k+1' and 'k-1' accesses would leave the grid, and the Fortran loop is 'DO k = ke,2,-1'
    accordingly. Rows 0 and nlev of both storages keep whatever this section found in them.

    Args:
        hhl: height of the model half levels [m] ('p_metrics%z_ifc'), nlev + 1 levels
        rhon: air density at half levels [kg/m3], as section 0) leaves it
        inverse_tke_time_step: 'fr_tke = 1 / dt_tke', the reciprocal TKE time step [1/s]

    Returns:
        inverse depth of the half-level layer [1/m], discretisation momentum of the TKE
        diffusion [kg/m2/s]
    """
    half = wpfloat("0.5")
    one = wpfloat("1.0")
    layer_depth = (hhl(Koff[-1]) - hhl(Koff[1])) * half
    inverse_layer_depth = one / layer_depth
    tke_discretisation_momentum = rhon * layer_depth * inverse_tke_time_step
    return inverse_layer_depth, tke_discretisation_momentum


@gtx.field_operator
def _gradient_of_conserved_variable(
    variable: fa.CellKField[wpfloat],
    inverse_depth: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Difference one quasi-conserved variable across the layer centred on a half level.

    One expression covers the whole column. The Fortran writes it twice because the reciprocal
    depth it divides by lives in two different arrays:

        zvari(i,ke1,n) = (zvari(i,ke,n)   - zvari(i,ke1,n)) * lays(i,ivtp(n))   ! at the surface
        zvari(i,k,n)   = (zvari(i,k-1,n)  - zvari(i,k,n))   * hlp(i,k)          ! above it

    Here the caller selects the reciprocal depth per row and this operator stays single-valued.
    """
    return (variable(Koff[-1]) - variable) * inverse_depth


@gtx.field_operator
def _compute_vertical_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Compute the vertical gradients of the five quasi-conserved model variables.

    Both of Raschendorfer's blocks evaluate the same difference quotient into the same storage;
    only the reciprocal depth differs, so they are one expression here and the row selection is
    a 'concat_where' on the depth.

    The five variables are the ones turbulence treats as dynamically active, in the Fortran's
    own order (mo_turbdiff_config.f90:62-77): the two horizontal wind components at the mass
    centre, then the three scalars conserved under condensation and evaporation -- liquid-water
    potential temperature, total water and liquid water. Section 0) put them on main levels;
    this section replaces them by their gradients on half levels, from which section 1b) builds
    the thermal and mechanical forcing of the TKE equation.

    THE TWO RECIPROCAL DEPTHS. Above the surface the difference spans a resolved layer, so it is
    divided by that layer's geometric depth and 'inverse_layer_depth' ('hlp', from the preceding
    statement of this program) is its reciprocal. The surface row of each variable instead holds
    its Prandtl-layer lower boundary value, which section 0) obtained from the second call of
    'adjust_satur_equil' (the winds from the transfer-layer reduction 'u(ke)*(1-tfm)'). That
    difference does not span a resolved layer; it is divided by the effective depth of the
    Prandtl layer, whose reciprocal is the surface transfer ratio from the first statement.
    Momentum and scalars have different Prandtl-layer resistances, hence two ratios; the Fortran
    selects between them with 'ivtp(n)', which maps the two wind components to 'mom' and
    'tet_l', 'h2o_g' and 'liq' to 'sca'.

    THE ROW-WISE OVERWRITE IS NOT A RECURRENCE. The Fortran writes each gradient back into the
    storage its variable came in, sweeping upward within each variable, so 'zvari(k-1)' is still
    the variable when 'zvari(k)' becomes the gradient; likewise the surface block runs before the
    interior one so that 'zvari(ke)' is still the variable when 'zvari(ke1)' is written. Every
    output row is therefore a function of input rows only, which is what makes this an ordinary
    'Koff[-1]' stencil rather than a scan (port spec 3.2). The sweep direction and the block order
    are what keep the aliasing safe in Fortran and mean nothing here, where inputs and outputs are
    separate fields.

    Row 0, the model top, is not written: there is no main level above the first one to difference
    against, which is why the Fortran loop stops at 'k = 2'.

    Args:
        zonal_wind: 'u_m', zonal wind at the mass centre [m/s], on main levels, with its
            Prandtl-layer boundary value in row 'nlev'
        meridional_wind: 'v_m', meridional wind at the mass centre [m/s], likewise
        liquid_water_potential_temperature: 'tet_l' [K], likewise
        total_water: 'h2o_g', specific total water content [kg/kg], likewise
        liquid_water: 'liq', specific liquid water content [kg/kg], likewise
        inverse_layer_depth: 'hlp', reciprocal depth of the half-level layer [1/m]; read only on
            rows 1 to 'nlev' - 1, which are the rows on which section 1a) defines it
        surface_transfer_ratio_for_momentum: 'lays(:,mom)', reciprocal effective Prandtl-layer
            depth for momentum [1/m]
        surface_transfer_ratio_for_scalars: 'lays(:,sca)', the same for scalars [1/m]
        nlev: 'ke', the index of the surface half level; the one row taking the Prandtl-layer
            depth instead of the geometric one

    Returns:
        the five vertical gradients on half levels: [1/s], [1/s], [K/m], [kg/kg/m], [kg/kg/m]
    """
    inverse_depth_for_momentum = concat_where(
        dims.KDim == nlev, surface_transfer_ratio_for_momentum, inverse_layer_depth
    )
    inverse_depth_for_scalars = concat_where(
        dims.KDim == nlev, surface_transfer_ratio_for_scalars, inverse_layer_depth
    )
    return (
        _gradient_of_conserved_variable(zonal_wind, inverse_depth_for_momentum),
        _gradient_of_conserved_variable(meridional_wind, inverse_depth_for_momentum),
        _gradient_of_conserved_variable(
            liquid_water_potential_temperature, inverse_depth_for_scalars
        ),
        _gradient_of_conserved_variable(total_water, inverse_depth_for_scalars),
        _gradient_of_conserved_variable(liquid_water, inverse_depth_for_scalars),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_gradients_of_conserved_variables(
    tvm: fa.CellField[wpfloat],
    tvh: fa.CellField[wpfloat],
    tkvm_at_surface: fa.CellField[wpfloat],
    tkvh_at_surface: fa.CellField[wpfloat],
    tfm: fa.CellField[wpfloat],
    tfh: fa.CellField[wpfloat],
    hhl: fa.CellKField[wpfloat],
    rhon: fa.CellKField[wpfloat],
    inverse_tke_time_step: wpfloat,
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
    tke_discretisation_momentum: fa.CellKField[wpfloat],
    zonal_wind_gradient: fa.CellKField[wpfloat],
    meridional_wind_gradient: fa.CellKField[wpfloat],
    liquid_water_potential_temperature_gradient: fa.CellKField[wpfloat],
    total_water_gradient: fa.CellKField[wpfloat],
    liquid_water_gradient: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Run the whole of section 1a): 'lays', then 'hlp' and 'dicke', then the five gradients.

    THE THREE VERTICAL RANGES, and where each comes from:

      * 'lays' has none. It is one value per column, Fortran 'DO i=ivstart,ivend' with no k-loop,
        so its statement's domain names 'CellDim' alone.
      * 'hlp' and 'dicke' run over rows 1..'nlev'-1, Fortran 'DO k=ke,2,-1'. The statement ends at
        'vertical_end - 1' because the surface half level has no layer centred on it.
      * The gradients run over rows 1..'nlev', Fortran 'DO k=ke,2,-1' plus the separate surface
        row, which is the row the 'concat_where' selects.

    'nlev' IS A SEPARATE ARGUMENT AND MUST STAY ONE, even though it equals 'vertical_end - 1'.
    'setup_program' inlines a scalar into the generated code, and the granule binds the domain
    bounds statically; making the 'concat_where' row index static at the same time FAILS TO COMPILE on
    'dace_cpu' -- the concat_where replacement pass asks a one-dimensional producer for a vertical
    offset it does not have. Measured 2026-08-28, and the surface branch here reads exactly such a
    one-dimensional producer. See 'Turbulence._program' for the traceback.

    Args:
        tvm: 'tvm', turbulent transfer velocity for momentum at the surface [m/s].
        tvh: 'tvh', the same for heat and moisture [m/s].
        tkvm_at_surface: 'tkvm(:,ke1)' [m2/s], taken BEFORE section 4) raises it.
        tkvh_at_surface: 'tkvh(:,ke1)' [m2/s], likewise.
        tfm: Prandtl-layer fraction of the transfer-layer resistance for momentum [1].
        tfh: the same for scalars [1].
        hhl: 'hhl' ('p_metrics%z_ifc'), height of the half levels [m].
        rhon: 'rhon', air density at half levels [kg/m3], as section 0) leaves it.
        inverse_tke_time_step: 'fr_tke = 1 / dt_tke' [1/s].
        zonal_wind: 'zvari(:,:,u_m)' [m/s] on main levels, surface row from section 0).
        meridional_wind: 'zvari(:,:,v_m)' [m/s], likewise.
        liquid_water_potential_temperature: 'zvari(:,:,tet_l)' [K], likewise.
        total_water: 'zvari(:,:,h2o_g)' [kg/kg], likewise.
        liquid_water: 'zvari(:,:,liq)' [kg/kg], likewise.
        nlev: 'ke', the surface half level; the row selected by the 'concat_where'.
        surface_transfer_ratio_for_momentum: Output, 'lays(:,mom)' [1/m]; read by the gradients.
        surface_transfer_ratio_for_scalars: Output, 'lays(:,sca)' [1/m]; likewise.
        inverse_layer_depth: Output, 'hlp' [1/m], rows 1..'nlev'-1; read by the gradients.
        tke_discretisation_momentum: Output, 'dicke' [kg/m2/s], rows 1..'nlev'-1.
        zonal_wind_gradient: Output, 'zvari(:,:,u_m)' as a gradient [1/s], rows 1..'nlev'.
        meridional_wind_gradient: Output, 'zvari(:,:,v_m)' [1/s], likewise.
        liquid_water_potential_temperature_gradient: Output, 'zvari(:,:,tet_l)' [K/m], likewise.
        total_water_gradient: Output, 'zvari(:,:,h2o_g)' [(kg/kg)/m], likewise.
        liquid_water_gradient: Output, 'zvari(:,:,liq)' [(kg/kg)/m], likewise.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. Row 0 is never written.
        vertical_end: End of the half levels for the GRADIENTS; 'nlev' + 1. 'hlp' and 'dicke'
            end one row earlier.
    """
    _compute_surface_transfer_ratios(
        tvm=tvm,
        tvh=tvh,
        tkvm_at_surface=tkvm_at_surface,
        tkvh_at_surface=tkvh_at_surface,
        tfm=tfm,
        tfh=tfh,
        out=(surface_transfer_ratio_for_momentum, surface_transfer_ratio_for_scalars),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
    _compute_inverse_layer_depth_and_tke_discretisation_momentum(
        hhl=hhl,
        rhon=rhon,
        inverse_tke_time_step=inverse_tke_time_step,
        out=(inverse_layer_depth, tke_discretisation_momentum),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _compute_vertical_gradients_of_conserved_variables(
        zonal_wind=zonal_wind,
        meridional_wind=meridional_wind,
        liquid_water_potential_temperature=liquid_water_potential_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        inverse_layer_depth=inverse_layer_depth,
        surface_transfer_ratio_for_momentum=surface_transfer_ratio_for_momentum,
        surface_transfer_ratio_for_scalars=surface_transfer_ratio_for_scalars,
        nlev=nlev,
        out=(
            zonal_wind_gradient,
            meridional_wind_gradient,
            liquid_water_potential_temperature_gradient,
            total_water_gradient,
            liquid_water_gradient,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

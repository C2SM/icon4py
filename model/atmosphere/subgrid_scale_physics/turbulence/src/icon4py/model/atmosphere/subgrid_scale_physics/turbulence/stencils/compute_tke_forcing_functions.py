# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Section 1b) of 'turbdiff': the two basic single-column forcing functions for TKE.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', from the
section whose banner reads "1b) Calculation of the basic single-column forcing functions for
TKE" (:1209-1240 at icon commit 26d6b98cce, which is the commit that produced the reference
capture; the section is :1196-1229 in the uninstrumented upstream file). The scientific
commentary in that file is by Matthias Raschendorfer (DWD).

The whole section is two ACC loops:

    DO k=2,ke1
      frh(i,k) = zaux(i,k,4)*zvari(i,k,tet_l) + zaux(i,k,5)*zvari(i,k,h2o_g)
    DO k=2,kem
      frm(i,k) = MAX( zvari(i,k,u_m)**2 + zvari(i,k,v_m)**2, fc_min(i) )

-- the buoyancy production and the pure single-column shear production of turbulent kinetic
energy. 'zaux(:,:,4)' and 'zaux(:,:,5)' are the 'g_tet' and 'g_h2o' outputs of
'adjust_satur_equil' (turb_utilities.f90), produced by section 0); the four gradients are
produced by section 1a); 'fc_min' comes from 'turb_setup'. All of them are half-level fields.

WHY THIS IS ONE PROGRAM AND NOT A 'concat_where'. The two forcings are different fields with
different formulas over vertical ranges that differ by one row, and fusing them into a single
output selection would need 'concat_where(KDim < nlev, shear, frm)' -- a read-modify-write that
turns "'frm(:,ke1)' is never written" into "'frm(:,ke1)' is rewritten with its old value". That
is a semantic change, and the package README rejects it on exactly those grounds.

None of it applies here, because a '@gtx.program' body is a SEQUENCE of field-operator calls,
each with its own 'out=' and its own 'domain='. The two statements below write precisely the
rows the two former programs wrote, in the same order and with the same expressions; 'frm(:,ke1)'
is not read, not written and not selected over. Nothing about the semantics moves -- one name
replaces two, and the name is the scheme's own.
"""

import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_thermal_forcing(
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    vertical_gradient_tet_l: fa.CellKField[wpfloat],
    vertical_gradient_h2o_g: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Buoyancy production of turbulent kinetic energy from the two conserved-variable gradients.

    'tet_l' is the liquid-water potential temperature and 'h2o_g' the total water content --
    Raschendorfer's names for the two variables that are conserved under condensation and
    evaporation, and therefore the pair the buoyancy flux is expressed in. Their buoyancy
    factors carry the moisture and cloud-cover dependence, so the sum below is the full moist
    buoyancy production and not a dry-air approximation.
    """
    return (
        buoyancy_factor_tet_l * vertical_gradient_tet_l
        + buoyancy_factor_h2o_g * vertical_gradient_h2o_g
    )


@gtx.field_operator
def _compute_mechanical_forcing(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Shear production of turbulent kinetic energy, floored by the minimal forcing.

    THE SQUARES ARE WRITTEN AS PRODUCTS, NOT AS '**2', and that is not a matter of taste.
    Fortran's 'x**2' with an integer literal exponent is a multiplication -- every compiler
    expands it -- while GT4Py lowers Python's 'x**2' to 'math.pow(x, 2)' and leaves it to the
    target's libm. Host libm gets that exactly right ('pow(x, 2.0)' is correctly rounded, and
    GCC folds it to 'x*x' anyway), but CUDA's 'pow' is a general-purpose implementation with a
    documented error of up to 2 ulp, so on the GPU backends 'x**2' is NOT 'x*x'. Measured on
    dace_gpu 2026-08-28: 'frm' differed from ICON by 1-2 ulp on about a quarter of the values,
    and 'test_compute_mechanical_forcing_is_the_fortran_expression_up_to_one_contraction' ruled
    out a fused multiply-add as the cause -- the difference matched none of the three admissible
    contractions. With the products below all four backends are bit-exact.
    """
    return maximum(
        vertical_gradient_u * vertical_gradient_u + vertical_gradient_v * vertical_gradient_v,
        min_forcing,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tke_forcing_functions(
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    vertical_gradient_tet_l: fa.CellKField[wpfloat],
    vertical_gradient_h2o_g: fa.CellKField[wpfloat],
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the thermal forcing 'frh' and the mechanical forcing 'frm' on half levels.

    THE TWO VERTICAL RANGES DIFFER BY ONE ROW, AND THE DIFFERENCE IS LOAD-BEARING. The thermal
    forcing runs to the surface half level 'ke1'; the mechanical forcing stops one half level
    higher, at 'kem = ke' (turb_diffusion.f90:843, "lowest model-layer, SUB 'turbdiff' is applied
    to"), which is why the second statement below ends at 'vertical_end - 1'.

    The note the Fortran leaves at that point (turb_diffusion.f90:1225) is the reason 'frh' goes
    one row further:

        "'frh' at "0"-level (k=ke1) is used for calculating the acceleration of non-turbulent
        near-surface circulations."

    and the local declaration (:783) agrees that the array is not always a forcing: "thermal
    forcing (1/s2) or thermal acceleration (m/s2)". Which later section consumes the 'ke1' row is
    not settled here -- section 6) overwrites the whole array with the circulation-kinetic-energy
    flux density -- but the asymmetry against 'frm' is deliberate, not an oversight.

    No statement anywhere in 'turbdiff' writes 'frm(:,ke1)', so that row keeps whatever the
    routine-local array was allocated with. It is in particular NOT the surface shear that
    'turbtran' computes with the same expression at turb_transfer.f90:954: that is a different
    local array of the same name in a different routine.

    THE FLOOR ON THE SHEAR. 'fc_min = (vel_min / MAX(l_hori, tur_len))**2' is set once per column
    in 'turb_setup' (turb_utilities.f90:338) and is the shear that a velocity scale of 'vel_min'
    over the effective horizontal length scale would produce. It is a physical floor on the
    forcing, not a division guard: nothing here divides by 'frm'. Raschendorfer records having
    tested its removal, at turb_utilities.f90:337 immediately above the assignment, alongside the
    commented-out alternative 'fc_min(i)=z0':

        "test: frm ohne fc_min-Beschraenkung: Bewirkt Unterschiede!"
        -- "test: 'frm' without the 'fc_min' restriction: causes differences!"

    which is why the floor is kept rather than dropped as a numerical nicety.

    Args:
        buoyancy_factor_tet_l: Buoyancy factor of the liquid-water potential temperature,
            'zaux(:,:,4)' = 'g_tet' [m/s2 per K], half levels.
        buoyancy_factor_h2o_g: Buoyancy factor of the total water content, 'zaux(:,:,5)' =
            'g_h2o' [m/s2 per (kg/kg)], half levels.
        vertical_gradient_tet_l: Vertical gradient of the liquid-water potential temperature,
            'zvari(:,:,tet_l)' [K/m], half levels.
        vertical_gradient_h2o_g: Vertical gradient of the total water content,
            'zvari(:,:,h2o_g)' [(kg/kg)/m], half levels.
        vertical_gradient_u: Vertical gradient of the zonal wind at the mass centre,
            'zvari(:,:,u_m)' [1/s], half levels.
        vertical_gradient_v: Vertical gradient of the meridional wind at the mass centre,
            'zvari(:,:,v_m)' [1/s], half levels.
        min_forcing: Lower limit of the TKE forcing, 'fc_min' [1/s2], one value per column.
        thermal_forcing: Output, 'frh' [1/s2], half levels; an acceleration [m/s2] at 'ke1'.
        mechanical_forcing: Output, 'frm' [1/s2], half levels; the surface half level 'ke1' is
            not written.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. Level 0 is never written,
            by either statement.
        vertical_end: End of the half levels for the THERMAL forcing; 'ke1', mirroring Fortran
            'k=...,ke1'. The mechanical forcing ends one row earlier, at 'kem = ke'.
    """
    _compute_thermal_forcing(
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        vertical_gradient_tet_l=vertical_gradient_tet_l,
        vertical_gradient_h2o_g=vertical_gradient_h2o_g,
        out=thermal_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_mechanical_forcing(
        vertical_gradient_u=vertical_gradient_u,
        vertical_gradient_v=vertical_gradient_v,
        min_forcing=min_forcing,
        out=mechanical_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

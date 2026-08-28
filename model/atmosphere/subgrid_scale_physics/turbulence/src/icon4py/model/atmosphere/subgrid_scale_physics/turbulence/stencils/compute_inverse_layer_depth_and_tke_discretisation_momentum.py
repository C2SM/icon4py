# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_inverse_layer_depth_and_tke_discretisation_momentum(
    hhl: fa.CellKField[wpfloat],
    rhon: fa.CellKField[wpfloat],
    inverse_tke_time_step: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the reciprocal half-level layer depth and the TKE discretisation momentum.

    Translated from ICON's turb_diffusion.f90, SUBROUTINE 'turbdiff', section
    "1a) Berechnung der benoetigten vertikalen Gradienten" ("Calculation of the required
    vertical gradients"), lines 1172-1184 at icon commit 26d6b98cce -- the block Matthias
    Raschendorfer heads "An den darueberliegenden Nebenflaechen" ("At the half levels above
    it", the ones above the lower boundary treated just before):

        wert       = (hhl(i,k-1) - hhl(i,k+1)) * z1d2
        hlp(i,k)   = z1 / wert
        dicke(i,k) = rhon(i,k) * wert * fr_tke

    'wert' is the depth of the layer centred on half level k, i.e. the distance between the
    two main levels that surround it, taken as half the distance between the enclosing half
    levels. It carries both quantities the rest of the scheme needs at a half level:

      * its reciprocal is the denominator of every centred vertical difference, and section
        1a) uses it immediately for the gradients of the quasi-conserved variables;
      * multiplied by the half-level density and divided by the TKE time step it is the
        discretisation momentum of the vertical TKE diffusion, 'rho_n * dz / dt_tke'
        [kg/m2/s] -- the mass per unit area of the layer, per unit time -- which section 9)
        hands to the semi-implicit solver as 'disc_mom'.

    The Fortran computes both from one 'wert' and so does this operator, so the two outputs
    round identically; the division and the two multiplications are in the Fortran's order.

    Neither output is defined at the model top (k = 0) or at the surface (k = nlev): the
    'k+1' and 'k-1' accesses would leave the grid, and the Fortran loop is 'DO k = ke,2,-1'
    accordingly. Run the program with 'vertical_start = 1', 'vertical_end = nlev'; rows 0 and
    nlev of both storages keep whatever this section found in them.

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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_layer_depth_and_tke_discretisation_momentum(
    hhl: fa.CellKField[wpfloat],
    rhon: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
    tke_discretisation_momentum: fa.CellKField[wpfloat],
    inverse_tke_time_step: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_inverse_layer_depth_and_tke_discretisation_momentum(
        hhl=hhl,
        rhon=rhon,
        inverse_tke_time_step=inverse_tke_time_step,
        out=(inverse_layer_depth, tke_discretisation_momentum),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

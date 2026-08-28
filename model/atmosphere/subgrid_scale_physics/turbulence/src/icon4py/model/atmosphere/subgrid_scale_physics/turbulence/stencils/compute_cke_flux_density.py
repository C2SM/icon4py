# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Scaled flux density of circulation kinetic energy at half levels.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
6) (:2118-2310 at icon commit 26d6b98cce), from the block Raschendorfer heads

    "Aufnahme des Zirkulationstermes mit Interpolation auf HF:"
    -- "Taking up the circulation term, with interpolation onto main levels"

at :2215-2217, first loop (:2221-2237). The scientific commentary in that file is by Matthias
Raschendorfer (DWD).
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_cke_flux_density(
    air_density: fa.CellKField[wpfloat],
    scalar_diffusion_coefficient: fa.CellKField[wpfloat],
    circulation_acceleration: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'rho_n * tkvh * a_circ * l', the length-scale-scaled CKE flux density.

    A plain product of four half-level fields, in the Fortran's order.
    """
    return air_density * scalar_diffusion_coefficient * circulation_acceleration * mixing_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cke_flux_density(
    air_density: fa.CellKField[wpfloat],
    scalar_diffusion_coefficient: fa.CellKField[wpfloat],
    circulation_acceleration: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
    cke_flux_density: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the scaled CKE flux density 'frh' on half levels (turb_diffusion.f90:2221-2237).

    The Fortran, with Raschendorfer's heading and trailing comment:

        ! Belegung von 'frh' mit der CKE-Flussdichte durch nicht-turbulente Zirkulationen, die
        !  durch thermische Inhomogenitaet an der Oberflaeche verursacht wird:
        frh(i,k) = rhon(i,k)*tkvh(i,k)*prss(i,k)*len_scale(i,k)   ! skalierte Flussdichte auf NF

        -- "Filling 'frh' with the flux density of circulation kinetic energy carried by the
           non-turbulent circulations that the thermal inhomogeneity of the surface causes"
           ... "scaled flux density at half levels"

    WHY THE FLUX IS SCALED BY THE LENGTH SCALE, from the note at :2231-2234:

        "'frh/len_scale' ist eine TKE-Flussdichte in [Kg/s3], deren Vertikalprofil im
         wesentlichen durch d_z(tet_v)**2 bestimmt ist, was zumindest in der Prandtl-Schicht
         prop. zu 1/len_scale ist. Die nachfolgende lineare Interpolation auf Hauptflaechen
         erfolgt daher mit 'frh'!"
        -- "'frh/len_scale' is a TKE flux density in [kg/s3] whose vertical profile is
           essentially determined by d_z(tet_v)**2, which -- at least within the Prandtl layer
           -- is proportional to 1/len_scale. The subsequent linear interpolation onto main
           levels is therefore performed on 'frh'!"

    In other words the factor 'len_scale' is not part of the physical flux; it is what makes
    the quantity smooth enough in the vertical that the linear interpolation of
    'compute_cke_flux_at_main_levels' is legitimate, and that operator divides it out again.

    THE STORAGE IS REUSED AND SO IS THE MEANING. 'frh' held the thermal (buoyancy) forcing of
    the TKE equation from section 1b) up to and including section 5); this section overwrites it
    with a flux density. Likewise 'circulation_acceleration' is the storage 'zvari(:,:,0)',
    which entered 'turbdiff' as the half-level air pressure ('prss => zvari(:,:,0)',
    turb_diffusion.f90:859) and which 'solve_turb_budgets' replaced in section 3) by the
    circulation acceleration proper, 'l_coh * fh2' (turb_utilities.f90:1728-1744). The Fortran
    name 'prss' at :2229 is therefore stale by two sections; it is the acceleration in [m/s2]
    that this product needs for its units to come out as [kg m/s3].

    The whole block is guarded by 'IF (lcircterm .OR. loutthcrc)', i.e. by 'pat_len > 0'
    (and 'ltkenst' for the first half). It is on in the reference capture, where
    'pat_len = 750 m'.

    Args:
        air_density: 'rhon' [kg/m3], half levels, as section 0) interpolated it.
        scalar_diffusion_coefficient: 'tkvh' [m2/s], half levels, as section 4) limited it.
        circulation_acceleration: 'prss' = 'zvari(:,:,0)' [m/s2], half levels: the vertical
            acceleration of the near-surface thermal circulations, as 'solve_turb_budgets' left
            it in section 3). NOT the pressure the Fortran name suggests.
        mixing_length: 'len_scale', the turbulent master length scale [m], half levels.
        cke_flux_density: Output, 'frh' [kg m/s3], half levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. The model top is not
            written.
        vertical_end: End of the half levels; 'ke1', mirroring Fortran 'k=...,ke1'. The surface
            half level IS written, and section 8) uses it.
    """
    _compute_cke_flux_density(
        air_density=air_density,
        scalar_diffusion_coefficient=scalar_diffusion_coefficient,
        circulation_acceleration=circulation_acceleration,
        mixing_length=mixing_length,
        out=cke_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

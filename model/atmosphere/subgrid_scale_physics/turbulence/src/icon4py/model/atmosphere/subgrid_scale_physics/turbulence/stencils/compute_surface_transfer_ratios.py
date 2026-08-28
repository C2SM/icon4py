# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
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

    Translated from ICON's turb_diffusion.f90, SUBROUTINE 'turbdiff', section
    "1a) Berechnung der benoetigten vertikalen Gradienten" ("Calculation of the required
    vertical gradients"), lines 1151-1160 at icon commit 26d6b98cce -- the block Matthias
    Raschendorfer heads "Am unteren Modellrand" ("At the lower boundary of the model"):

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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_transfer_ratios(
    tvm: fa.CellField[wpfloat],
    tvh: fa.CellField[wpfloat],
    tkvm_at_surface: fa.CellField[wpfloat],
    tkvh_at_surface: fa.CellField[wpfloat],
    tfm: fa.CellField[wpfloat],
    tfh: fa.CellField[wpfloat],
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
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

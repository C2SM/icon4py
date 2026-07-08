# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_wp import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_shear_to_half_level_cells(
    shear: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Interpolate the rate of strain (mechanical production term, 2 * |S|^2) from
    edge midpoints of full levels to cell centers at half levels.

    Port of ``interpolate_rate_of_strain_full2half_edge2cell`` in ICON's
    ``mo_vdf_atmo.f90``: the shear is first averaged from the three C2E
    neighbor edges to the cell center with the bilinear weights ``e_bln_c_s``
    (separately on full levels k and k - 1), then interpolated vertically to
    the half level k with the weights ``wgtfac_c`` (reuses the common field
    operator ``_interpolate_cell_field_to_half_levels_wp``):

        mech_prod(c, k) = wgtfac_c(c, k) * sum_e e_bln_c_s(c, e) * shear(e, k)
                          + (1 - wgtfac_c(c, k))
                            * sum_e e_bln_c_s(c, e) * shear(e, k - 1)

    The Fortran loop runs over jk = 2..nlev (1-based), i.e. half levels
    k = 1..nlev-1 (0-based); the top (k = 0) and bottom (k = nlev) half-level
    rows are not computed. The call site (``Compute_diagnostics`` in
    ``mo_vdf_atmo.f90``) uses ``rl_start = 3`` and
    ``rl_end = min_rlcell_int - 1``, i.e. ``h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3``
    to ``h_grid.Zone.HALO`` for cells.
    """
    shear_c = neighbor_sum(e_bln_c_s * shear(C2E), axis=C2EDim)
    return _interpolate_cell_field_to_half_levels_wp(wgtfac_c=wgtfac_c, interpolant=shear_c)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_shear_to_half_level_cells(
    shear: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    mech_prod: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_shear_to_half_level_cells(
        shear=shear,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        out=mech_prod,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

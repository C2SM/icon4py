# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_wp import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_cell_to_half_levels(
    interpolant: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    wgtfacq1_c_1: fa.CellField[wpfloat],
    wgtfacq1_c_2: fa.CellField[wpfloat],
    wgtfacq1_c_3: fa.CellField[wpfloat],
    wgtfacq_c_1: fa.CellField[wpfloat],
    wgtfacq_c_2: fa.CellField[wpfloat],
    wgtfacq_c_3: fa.CellField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """
    Interpolate a cell field from full levels to half levels, including the
    extrapolated top and bottom boundary half levels.

    Port of ``vert_intp_full2half_cell_3d`` in ICON's
    ``mo_nh_vert_interp_les.f90``:
    - interior half levels (0 < k < nlev, Fortran jk = 2..nlev) are linearly
      interpolated with ``wgtfac_c`` (reuses the common field operator
      ``_interpolate_cell_field_to_half_levels_wp``),
    - the top half level (k == 0, Fortran jk = 1) is extrapolated quadratically
      from the first three full levels with ``wgtfacq1_c_1/2/3``,
    - the bottom half level (k == nlev, Fortran jk = nlevp1) is extrapolated
      quadratically from the last three full levels with ``wgtfacq_c_1/2/3``.

    Each ``concat_where`` branch is evaluated only on its own K region, so the
    vertical (``Koff``) shifts in the branch expressions need to be in bounds
    only there.

    The tmx call site (rho -> rho_ic in ``Compute_diagnostics`` of
    ``mo_vdf_atmo.f90``) uses ``rl_start = 2``, ``rl_end = min_rlcell_int - 2``,
    which maps to the horizontal domain
    ``(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, h_grid.Zone.HALO_LEVEL_2)``.

    Args:
        interpolant: cell field on full levels (nlev levels)
        wgtfac_c: interpolation weight on half levels
        wgtfacq1_c_1: top extrapolation weight for full level 0
        wgtfacq1_c_2: top extrapolation weight for full level 1
        wgtfacq1_c_3: top extrapolation weight for full level 2
        wgtfacq_c_1: bottom extrapolation weight for full level nlev - 1
        wgtfacq_c_2: bottom extrapolation weight for full level nlev - 2
        wgtfacq_c_3: bottom extrapolation weight for full level nlev - 3
        nlev: number of full levels

    Returns:
        cell field on half levels (nlev + 1 levels)
    """
    interpolation_interior = _interpolate_cell_field_to_half_levels_wp(
        wgtfac_c=wgtfac_c, interpolant=interpolant
    )
    interpolation_top = (
        wgtfacq1_c_1 * interpolant
        + wgtfacq1_c_2 * interpolant(KDim + 1)
        + wgtfacq1_c_3 * interpolant(KDim + 2)
    )
    interpolation_bottom = (
        wgtfacq_c_1 * interpolant(KDim - 1)
        + wgtfacq_c_2 * interpolant(KDim - 2)
        + wgtfacq_c_3 * interpolant(KDim - 3)
    )
    interpolation = concat_where(dims.KDim == 0, interpolation_top, interpolation_interior)
    return concat_where(dims.KDim == nlev, interpolation_bottom, interpolation)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_cell_to_half_levels(
    interpolant: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    wgtfacq1_c_1: fa.CellField[wpfloat],
    wgtfacq1_c_2: fa.CellField[wpfloat],
    wgtfacq1_c_3: fa.CellField[wpfloat],
    wgtfacq_c_1: fa.CellField[wpfloat],
    wgtfacq_c_2: fa.CellField[wpfloat],
    wgtfacq_c_3: fa.CellField[wpfloat],
    interpolation: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_cell_to_half_levels(
        interpolant=interpolant,
        wgtfac_c=wgtfac_c,
        wgtfacq1_c_1=wgtfacq1_c_1,
        wgtfacq1_c_2=wgtfacq1_c_2,
        wgtfacq1_c_3=wgtfacq1_c_3,
        wgtfacq_c_1=wgtfacq_c_1,
        wgtfacq_c_2=wgtfacq_c_2,
        wgtfacq_c_3=wgtfacq_c_3,
        nlev=nlev,
        out=interpolation,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

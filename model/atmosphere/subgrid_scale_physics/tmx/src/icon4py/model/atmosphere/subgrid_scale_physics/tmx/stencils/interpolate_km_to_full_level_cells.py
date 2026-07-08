# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vertical_operations import average_level_plus1_on_cells
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_km_to_full_level_cells(
    km_ic: fa.CellKField[wpfloat],
    km_min: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Interpolate the eddy viscosity from half-level cell centers to full-level
    cell centers and apply the minimum-viscosity floor:

        km_c(k) = max(km_min, 0.5 * (km_ic(k) + km_ic(k + 1)))

    Port of ``interpolate_eddy_viscosity2cell`` in ICON's ``mo_vdf_atmo.f90``
    (reuses the common field operator ``average_level_plus1_on_cells``).
    The floor deliberately lives here (and in the vertex/edge interpolations)
    and not in the Smagorinsky viscosity computation, matching the Fortran.

    Vertical: Fortran ``jk = 1, nlev`` (1-based) -> full levels k = 0..nlev-1
    (0-based); ``km_ic`` has nlev + 1 half levels, so ``km_ic(KDim + 1)`` is in
    bounds on the whole output domain.

    Horizontal (call site in ``Compute_diagnostics``): ``rl_start =
    grf_bdywidth_c`` -> ``h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4`` (cells),
    ``rl_end = min_rlcell_int - 1`` -> ``h_grid.Zone.HALO`` (cells); halo cells
    are computed on purpose because ``km_c`` is used in the diffusion later.

    Args:
        km_ic: eddy viscosity at half-level cell centers (nlev + 1 levels)
        km_min: minimum eddy viscosity

    Returns:
        eddy viscosity at full-level cell centers (nlev levels)
    """
    return maximum(km_min, average_level_plus1_on_cells(km_ic))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km_to_full_level_cells(
    km_ic: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    km_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_km_to_full_level_cells(
        km_ic=km_ic,
        km_min=km_min,
        out=km_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

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
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_km_to_vertices(
    km_ic: fa.CellKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    km_min: wpfloat,
) -> fa.VertexKField[wpfloat]:
    """
    Interpolate the eddy viscosity from half-level cell centers to half-level
    vertices (area-weighted V2C gather with ``cells_aw_verts``) and apply the
    minimum-viscosity floor:

        km_iv = max(km_min, sum_{c in V2C} cells_aw_verts * km_ic(c))

    Port of ``interpolate_eddy_viscosity2half_vertex`` in ICON's
    ``mo_vdf_atmo.f90`` (``cells2verts_scalar`` with ``ptr_int%cells_aw_verts``,
    reusing the common field operator ``_compute_cell_2_vertex_interpolation``,
    followed by the ``MAX(km_min, ...)`` loop), with the floor fused into the
    interpolation.

    Vertical: ``cells2verts_scalar`` defaults ``slev = 1``, ``elev =
    UBOUND(km_ic, 2)`` -> all half levels, k = 0..nlev (0-based, nlev + 1 rows).

    Horizontal: ``opt_rlstart = 5 (= max_rlvert)`` -> ``h_grid.Zone.NUDGING``
    (vertices), ``opt_rlend = min_rlvert_int - 1`` -> ``h_grid.Zone.HALO``
    (vertices); halo vertices are computed on purpose because ``km_iv`` is used
    in the diffusion later.

    Note: the Fortran applies the ``MAX(km_min, ...)`` floor to the *entire*
    ``km_iv`` array (which was initialized to zero beforehand), so vertices
    outside the interpolated region end up holding ``km_min``. Here the floor
    is fused with the gather and only acts on the program domain; the caller
    must initialize ``km_iv`` to ``km_min`` (instead of zero) if values outside
    this domain are ever read.

    Args:
        km_ic: eddy viscosity at half-level cell centers (nlev + 1 levels)
        cells_aw_verts: cell-to-vertex area-weighted interpolation coefficients
        km_min: minimum eddy viscosity

    Returns:
        eddy viscosity at half-level vertices (nlev + 1 levels)
    """
    return maximum(km_min, _compute_cell_2_vertex_interpolation(km_ic, cells_aw_verts))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km_to_vertices(
    km_ic: fa.CellKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    km_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_km_to_vertices(
        km_ic=km_ic,
        cells_aw_verts=cells_aw_verts,
        km_min=km_min,
        out=km_iv,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

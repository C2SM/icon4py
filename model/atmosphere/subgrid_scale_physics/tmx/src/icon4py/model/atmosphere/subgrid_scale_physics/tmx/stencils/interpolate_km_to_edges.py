# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_km_to_edges(
    km_ic: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    km_min: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """
    Interpolate the eddy viscosity from half-level cell centers to half-level
    edges (linear E2C gather with ``c_lin_e``) and apply the minimum-viscosity
    floor:

        km_ie = max(km_min, sum_{c in E2C} c_lin_e * km_ic(c))

    Port of ``interpolate_eddy_viscosity2half_edge`` in ICON's
    ``mo_vdf_atmo.f90`` (``cells2edges_scalar`` with ``ptr_int%c_lin_e``
    followed by the ``MAX(km_min, ...)`` loop), with the floor fused into the
    interpolation. The single-neighbor lateral-boundary fill of
    ``cells2edges_scalar`` (edges with ``refin_ctrl`` 1..2) is not reached at
    this call site (``opt_rlstart = grf_bdywidth_e``) and is not ported.

    Vertical: ``cells2edges_scalar`` defaults ``slev = 1``, ``elev =
    UBOUND(km_ic, 2)`` -> all half levels, k = 0..nlev (0-based, nlev + 1 rows).

    Horizontal: ``opt_rlstart = grf_bdywidth_e (= 9)`` -> ``h_grid.Zone.NUDGING``
    (edges), ``opt_rlend = min_rledge_int - 1`` -> ``h_grid.Zone.HALO`` (edges);
    halo edges are computed on purpose because ``km_ie`` is used in the
    diffusion later.

    Note: the Fortran applies the ``MAX(km_min, ...)`` floor to the *entire*
    ``km_ie`` array (which was initialized to zero beforehand), so edges outside
    the interpolated region end up holding ``km_min``. Here the floor is fused
    with the gather and only acts on the program domain; the caller must
    initialize ``km_ie`` to ``km_min`` (instead of zero) if values outside this
    domain are ever read.

    Args:
        km_ic: eddy viscosity at half-level cell centers (nlev + 1 levels)
        c_lin_e: cell-to-edge linear interpolation coefficients
        km_min: minimum eddy viscosity

    Returns:
        eddy viscosity at half-level edges (nlev + 1 levels)
    """
    return maximum(km_min, neighbor_sum(km_ic(E2C) * c_lin_e, axis=E2CDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_km_to_edges(
    km_ic: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    km_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_km_to_edges(
        km_ic=km_ic,
        c_lin_e=c_lin_e,
        km_min=km_min,
        out=km_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

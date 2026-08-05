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
from icon4py.model.common.dimension import E2C2E, E2C2EDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_tangential_wind_wp(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Reconstruct the tangential velocity component at edge midpoints from the
    normal velocity components of the four surrounding edges (RBF vector
    interpolation).

    Working-precision port of ``rbf_vec_interpol_edge`` in ICON's
    ``mo_intp_rbf.f90`` (``rbf_vec_interpol_edge_lib`` in iconmath's
    ``mo_lib_intp_rbf.F90``):

        vt(e, k) = sum over the four E2C2E neighbor edges e' of
                   rbf_vec_coeff_e(e, e') * vn(e', k)

    The stencil is agnostic of the vertical staggering: tmx uses it both on
    half-level input (``vn_ie -> vt_ie``, Stage A) and on full-level input
    (``vn -> vt``, Stage E1); the K extent is just a domain argument.

    The Fortran tmx call site (``Compute_diagnostics`` in ``mo_vdf_atmo.f90``)
    uses ``opt_rlstart = 3`` and ``opt_rlend = min_rledge_int - 2``, i.e.
    ``h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3`` to ``h_grid.Zone.HALO_LEVEL_2``
    for edges, on all levels of the input field.
    """
    return neighbor_sum(rbf_vec_coeff_e * vn(E2C2E), axis=E2C2EDim)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tangential_wind_wp(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    vt: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tangential_wind_wp(
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        out=vt,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

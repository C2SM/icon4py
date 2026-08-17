# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat


# TODO(dastrm): this stencil has no test


sys.setrecursionlimit(5500)

#: Slot value for "this patch is empty". The gather that consumes it still reads some cell,
#: exactly as ICON's index 0 would, and the caller masks the contribution with famask; the
#: departure region area is zero there.
#:
#: The gtfn backend does not fold a module-level constant referenced inside a field operator
#: into the IR ("Symbols not found"), the same limitation _WENO_EPS carries in
#: accumulate_weno_candidate_flux_weights, so the field operator below inlines this literal
#: and callers and tests import this name, keeping one definition of the value.
_NO_PATCH = -1


@gtx.field_operator
def _compute_ffsl_flux_area_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    bf_cc_patch1_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch1_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[gtx.int32],
    fa.EdgeKField[gtx.int32],
]:
    famask_bool = famask_int == 1
    lvn_pos = p_vn >= 0.0
    # Translation of patch 1 and patch 2 in system relative to respective cell
    bf_cc_patch1_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lon[dims.E2CDim(0)], bf_cc_patch1_lon[dims.E2CDim(1)]),
        0.0,
    )
    bf_cc_patch1_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lat[dims.E2CDim(0)], bf_cc_patch1_lat[dims.E2CDim(1)]),
        0.0,
    )
    bf_cc_patch2_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lon[dims.E2CDim(0)], bf_cc_patch2_lon[dims.E2CDim(1)]),
        0.0,
    )
    bf_cc_patch2_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lat[dims.E2CDim(0)], bf_cc_patch2_lat[dims.E2CDim(1)]),
        0.0,
    )

    # patch1 in translated system
    dreg_patch1_1_lon_vmask = dreg_patch1_1_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_1_lat_vmask = dreg_patch1_1_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    dreg_patch1_2_lon_vmask = dreg_patch1_2_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_2_lat_vmask = dreg_patch1_2_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    dreg_patch1_3_lon_vmask = dreg_patch1_3_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_3_lat_vmask = dreg_patch1_3_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    dreg_patch1_4_lon_vmask = dreg_patch1_4_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_4_lat_vmask = dreg_patch1_4_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    # patch2 in translated system
    dreg_patch2_1_lon_vmask = dreg_patch2_1_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_1_lat_vmask = dreg_patch2_1_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)
    dreg_patch2_2_lon_vmask = dreg_patch2_2_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_2_lat_vmask = dreg_patch2_2_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)
    dreg_patch2_3_lon_vmask = dreg_patch2_3_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_3_lat_vmask = dreg_patch2_3_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)
    dreg_patch2_4_lon_vmask = dreg_patch2_4_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_4_lat_vmask = dreg_patch2_4_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)

    # Which butterfly cell each outer patch lies in, as a slot into the E2C2E2C offset.
    #
    # The bf_cc_patch* inputs above pin the same mapping from the other side, which is a
    # useful cross-check when wiring the rest of FFSL: they come from
    # pos_on_tplane_c_edge(:,:,side,4:5) (mo_advection_geometry.f90:209), whose last axis is
    # the shared vertex, so bf_cc_patch1[E2CDim(side)] is the vertex-0 butterfly centre on
    # that side and bf_cc_patch2[E2CDim(side)] the vertex-1 one. In slot terms:
    #   bf_cc_patch1[E2CDim(0)] -> slot 0    bf_cc_patch1[E2CDim(1)] -> slot 2
    #   bf_cc_patch2[E2CDim(0)] -> slot 1    bf_cc_patch2[E2CDim(1)] -> slot 3
    # ICON stores the absolute cell index (butterfly_idx) and picks it with
    # MERGE(butterfly_idx(je,jb,1,p), butterfly_idx(je,jb,2,p), lvn_pos), f90 799-803. Since
    # the slot is 2 * side + vertex, and patch 1 is the vertex-0 wing while patch 2 is the
    # vertex-1 wing, that MERGE is just a choice of side, so the whole lookup collapses to
    # two constants and the eight index/block input fields disappear.
    patch1_cell_slot_vmask = where(famask_bool, where(lvn_pos, 0, 2), -1)  # -1 = _NO_PATCH
    patch2_cell_slot_vmask = where(famask_bool, where(lvn_pos, 1, 3), -1)  # -1 = _NO_PATCH

    return (
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
        dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask,
        patch1_cell_slot_vmask,
        patch2_cell_slot_vmask,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ffsl_flux_area_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    bf_cc_patch1_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch1_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    patch1_cell_slot_vmask: fa.EdgeKField[gtx.int32],
    patch2_cell_slot_vmask: fa.EdgeKField[gtx.int32],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ffsl_flux_area_list(
        famask_int=famask_int,
        p_vn=p_vn,
        bf_cc_patch1_lon=bf_cc_patch1_lon,
        bf_cc_patch1_lat=bf_cc_patch1_lat,
        bf_cc_patch2_lon=bf_cc_patch2_lon,
        bf_cc_patch2_lat=bf_cc_patch2_lat,
        dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
        dreg_patch2_1_lon_vmask=dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask=dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask=dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask=dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask=dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask=dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask=dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask=dreg_patch2_4_lat_vmask,
        out=(
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
            patch1_cell_slot_vmask,
            patch2_cell_slot_vmask,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

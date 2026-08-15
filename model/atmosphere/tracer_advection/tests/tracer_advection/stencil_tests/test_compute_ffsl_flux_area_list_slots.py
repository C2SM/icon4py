"""The butterfly slot that compute_ffsl_flux_area_list emits for the two outer patches.

ICON stores the absolute cell index and selects it with
MERGE(butterfly_idx(je,jb,1,p), butterfly_idx(je,jb,2,p), lvn_pos)
(mo_advection_geometry.f90:799-803). Since the E2C2E2C slot is 2 * side + vertex, and patch 1
is the vertex-0 wing while patch 2 is the vertex-1 wing, that MERGE reduces to a choice of
side. This pins that reduction, which is the whole substitution: if the sign convention were
backwards, or the patch-to-vertex mapping swapped, FFSL would gather the wrong cell and still
produce plausible numbers.
"""

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection.stencils.compute_ffsl_flux_area_list import (
    _NO_PATCH,
    compute_ffsl_flux_area_list,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc


_DREG_ARGS = tuple(
    f"dreg_patch{patch}_{vertex}_{coord}_vmask"
    for patch in (1, 2)
    for vertex in (1, 2, 3, 4)
    for coord in ("lon", "lat")
)


@pytest.mark.parametrize(
    ("famask", "vn", "expected_patch1_slot", "expected_patch2_slot"),
    [
        # vn > 0: the departure region is on the e2c[0] side, so side = 0
        (1, 1.0, 0, 1),
        # vn < 0: side = 1
        (1, -1.0, 2, 3),
        # no flux area: no patch, whatever the sign
        (0, 1.0, _NO_PATCH, _NO_PATCH),
        (0, -1.0, _NO_PATCH, _NO_PATCH),
    ],
)
def test_patch_slots_follow_the_vn_sign(
    *,
    famask: int,
    vn: float,
    expected_patch1_slot: int,
    expected_patch2_slot: int,
    grid: base.Grid,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    def edge_k(value: float = 0.0) -> gtx.Field:
        return data_alloc.constant_field(
            grid, value, dims.EdgeDim, dims.KDim, allocator=allocator
        )

    inputs = {
        "famask_int": data_alloc.constant_field(
            grid, famask, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        ),
        "p_vn": edge_k(vn),
        **{
            name: data_alloc.zero_field(
                grid, dims.EdgeDim, dims.E2CDim, allocator=allocator
            )
            for name in (
                "bf_cc_patch1_lon",
                "bf_cc_patch1_lat",
                "bf_cc_patch2_lon",
                "bf_cc_patch2_lat",
            )
        },
        **{name: edge_k() for name in _DREG_ARGS},
    }
    patch1_slot = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
    )
    patch2_slot = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
    )

    compute_ffsl_flux_area_list.with_backend(backend)(
        **inputs,
        patch1_cell_slot_vmask=patch1_slot,
        patch2_cell_slot_vmask=patch2_slot,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(grid.num_edges),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(grid.num_levels),
        offset_provider=grid.connectivities,
    )

    assert np.unique(patch1_slot.asnumpy()).tolist() == [expected_patch1_slot]
    assert np.unique(patch2_slot.asnumpy()).tolist() == [expected_patch2_slot]

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import neighbor_sum
from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import E2CDim

from icon4pytools.common.metadata import get_stencil_info
from icon4pytools.icon4pygen.bindings.workflow import PyBindGen


def test_horizontal_field_sid_rendering():
    @field_operator
    def identity(field: gtx.Field[[dims.EdgeDim], float]) -> gtx.Field[[dims.EdgeDim], float]:
        return field

    @program
    def identity_prog(
        field: gtx.Field[[dims.EdgeDim], float], out: gtx.Field[[dims.EdgeDim], float]
    ):
        identity(field, out=out)

    stencil_info = get_stencil_info(identity_prog)
    bindgen = PyBindGen(stencil_info, 128, 1)
    for field in bindgen.fields:
        assert (
            field.renderer.render_sid()
            == "gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1)"
        )


def test_vertical_field_sid_rendering():
    @field_operator
    def identity(field: gtx.Field[[dims.KDim], float]) -> gtx.Field[[dims.KDim], float]:
        return field

    @program
    def identity_prog(field: gtx.Field[[dims.KDim], float], out: gtx.Field[[dims.KDim], float]):
        identity(field, out=out)

    stencil_info = get_stencil_info(identity_prog)
    bindgen = PyBindGen(stencil_info, 128, 1)
    for field in bindgen.fields:
        assert (
            field.renderer.render_sid()
            == "gridtools::hymap::keys<unstructured::dim::vertical>::make_values(1)"
        )


def test_dense_field_sid_rendering():
    @field_operator
    def identity(
        field: gtx.Field[[dims.EdgeDim, dims.KDim], float],
    ) -> gtx.Field[[dims.EdgeDim, dims.KDim], float]:
        return field

    @program
    def identity_prog(
        field: gtx.Field[[dims.EdgeDim, dims.KDim], float],
        out: gtx.Field[[dims.EdgeDim, dims.KDim], float],
    ):
        identity(field, out=out)

    stencil_info = get_stencil_info(identity_prog)
    bindgen = PyBindGen(stencil_info, 128, 1)
    for field in bindgen.fields:
        assert (
            field.renderer.render_sid()
            == "gridtools::hymap::keys<unstructured::dim::horizontal,unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride)"
        )


def test_vertical_sparse_field_sid_rendering():
    @field_operator
    def reduction(
        nb_field: gtx.Field[[dims.EdgeDim, E2CDim, dims.KDim], float],
    ) -> gtx.Field[[dims.EdgeDim, dims.KDim], float]:
        return neighbor_sum(nb_field, axis=E2CDim)

    @program
    def reduction_prog(
        nb_field: gtx.Field[[dims.EdgeDim, E2CDim, dims.KDim], float],
        out: gtx.Field[[dims.EdgeDim, dims.KDim], float],
    ):
        reduction(nb_field, out=out)

    stencil_info = get_stencil_info(reduction_prog)
    bindgen = PyBindGen(stencil_info, 128, 1)
    for field in bindgen.fields:
        assert (
            field.renderer.render_sid()
            == "gridtools::hymap::keys<unstructured::dim::horizontal,unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride)"
        )

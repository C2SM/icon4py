# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum
from icon4py.model.common.dimension import E2CDim, EdgeDim, KDim

from icon4pytools.icon4pygen.bindings.workflow import PyBindGen
from icon4pytools.icon4pygen.metadata import get_stencil_info


def test_horizontal_field_sid_rendering():
    @field_operator
    def identity(field: Field[[EdgeDim], float]) -> Field[[EdgeDim], float]:
        return field

    @program
    def identity_prog(field: Field[[EdgeDim], float], out: Field[[EdgeDim], float]):
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
    def identity(field: Field[[KDim], float]) -> Field[[KDim], float]:
        return field

    @program
    def identity_prog(field: Field[[KDim], float], out: Field[[KDim], float]):
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
    def identity(field: Field[[EdgeDim, KDim], float]) -> Field[[EdgeDim, KDim], float]:
        return field

    @program
    def identity_prog(field: Field[[EdgeDim, KDim], float], out: Field[[EdgeDim, KDim], float]):
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
    def reduction(nb_field: Field[[EdgeDim, E2CDim, KDim], float]) -> Field[[EdgeDim, KDim], float]:
        return neighbor_sum(nb_field, axis=E2CDim)

    @program
    def reduction_prog(
        nb_field: Field[[EdgeDim, E2CDim, KDim], float],
        out: Field[[EdgeDim, KDim], float],
    ):
        reduction(nb_field, out=out)

    stencil_info = get_stencil_info(reduction_prog)
    bindgen = PyBindGen(stencil_info, 128, 1)
    for field in bindgen.fields:
        assert (
            field.renderer.render_sid()
            == "gridtools::hymap::keys<unstructured::dim::horizontal,unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride)"
        )

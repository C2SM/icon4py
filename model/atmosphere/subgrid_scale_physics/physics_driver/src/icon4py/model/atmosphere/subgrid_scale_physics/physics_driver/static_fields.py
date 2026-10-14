# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The static fields of the physics driver and their construction from the field factories."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import geometry_attributes
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes


if TYPE_CHECKING:
    from icon4py.model.common.states import factory


@dataclasses.dataclass(frozen=True)
class PhysicsDriverStaticState:
    """The grid fields the physics driver's dyn2phy and phy2dyn stencils read."""

    ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Layer thickness at cell centers on full levels [m]."""
    rbf_vec_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]  # pyright: ignore[reportInvalidTypeForm]
    """RBF coefficients for the zonal wind component at cell centers."""
    rbf_vec_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]  # pyright: ignore[reportInvalidTypeForm]
    """RBF coefficients for the meridional wind component at cell centers."""
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]  # pyright: ignore[reportInvalidTypeForm]
    """Zonal component of the edge normal at the neighbouring cell centers."""
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]  # pyright: ignore[reportInvalidTypeForm]
    """Meridional component of the edge normal at the neighbouring cell centers."""
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]  # pyright: ignore[reportInvalidTypeForm]
    """Coefficients for linear interpolation from cell centers to edge midpoints."""


def build_static_state(
    *,
    geometry_source: factory.FieldSource,
    interpolation_source: factory.FieldSource,
    metrics_source: factory.FieldSource,
) -> PhysicsDriverStaticState:
    return PhysicsDriverStaticState(
        ddqz_z_full=metrics_source.get(metrics_attributes.DDQZ_Z_FULL),
        rbf_vec_coeff_c1=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_C1),
        rbf_vec_coeff_c2=interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_C2),
        primal_normal_cell_x=geometry_source.get(geometry_attributes.EDGE_NORMAL_CELL_U),
        primal_normal_cell_y=geometry_source.get(geometry_attributes.EDGE_NORMAL_CELL_V),
        c_lin_e=interpolation_source.get(interpolation_attributes.C_LIN_E),
    )

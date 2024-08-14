# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools

import gt4py.next as gtx

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import (
    C2E2CODim,
    CECDim,
    CEDim,
    CellDim,
    KDim,
    V2EDim,
    VertexDim,
)


@dataclasses.dataclass(frozen=True)
class DiffusionDiagnosticState:
    """Represents the diagnostic fields needed in diffusion."""

    # fields for 3D elements in turbdiff
    hdef_ic: fa.CellKField[float]  # ! divergence at half levels(nproma,nlevp1,nblks_c)     [1/s]
    div_ic: fa.CellKField[
        float
    ]  # ! horizontal wind field deformation (nproma,nlevp1,nblks_c)     [1/s^2]
    dwdx: fa.CellKField[
        float
    ]  # zonal gradient of vertical wind speed (nproma,nlevp1,nblks_c)     [1/s]
    dwdy: fa.CellKField[float]  # meridional gradient of vertical wind speed (nproma,nlevp1,nblks_c)


@dataclasses.dataclass(frozen=True)
class DiffusionMetricState:
    """Represents the metric state fields needed in diffusion."""

    theta_ref_mc: fa.CellKField[float]
    wgtfac_c: fa.CellKField[
        float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    mask_hdiff: fa.CellKField[bool]
    zd_vertoffset: gtx.Field[[CECDim, KDim], gtx.int32]
    zd_diffcoef: fa.CellKField[float]
    zd_intcoef: gtx.Field[[CECDim, KDim], float]


@dataclasses.dataclass(frozen=True)
class DiffusionInterpolationState:
    """Represents the ICON interpolation state needed in diffusion."""

    e_bln_c_s: gtx.Field[
        [CEDim], float
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[[CEDim], float]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[
        [CellDim, C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[[CellDim, C2E2CODim], float]
    geofac_grg_y: gtx.Field[
        [CellDim, C2E2CODim], float
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: fa.EdgeField[float]  # Nudgeing coeffients for edges

    @functools.cached_property
    def geofac_n2s_c(self) -> fa.CellField[float]:
        return gtx.as_field((CellDim,), data=self.geofac_n2s.ndarray[:, 0])

    @functools.cached_property
    def geofac_n2s_nbh(self) -> gtx.Field[[CECDim], float]:
        geofac_nbh_ar = self.geofac_n2s.ndarray[:, 1:]
        old_shape = geofac_nbh_ar.shape
        return gtx.as_field(
            (CECDim,),
            geofac_nbh_ar.reshape(
                old_shape[0] * old_shape[1],
            ),
        )

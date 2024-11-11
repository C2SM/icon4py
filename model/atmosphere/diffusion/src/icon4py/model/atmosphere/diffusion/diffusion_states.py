# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import functools
from dataclasses import dataclass

from gt4py.next import as_field
from gt4py.next.common import Dimension, Field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.settings import xp
from icon4py.model.common.grid.base import BaseGrid

from icon4py.model.common.dimension import (
    C2E2CODim,
    CECDim,
    CEDim,
    CellDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)


@dataclass(frozen=True)
class DiffusionDiagnosticState:
    """Represents the diagnostic fields needed in diffusion."""

    # fields for 3D elements in turbdiff
    hdef_ic: Field[
        [CellDim, KDim], float
    ]  # ! divergence at half levels(nproma,nlevp1,nblks_c)     [1/s]
    div_ic: Field[
        [CellDim, KDim], float
    ]  # ! horizontal wind field deformation (nproma,nlevp1,nblks_c)     [1/s^2]
    dwdx: Field[
        [CellDim, KDim], float
    ]  # zonal gradient of vertical wind speed (nproma,nlevp1,nblks_c)     [1/s]

    dwdy: Field[
        [CellDim, KDim], float
    ]  # meridional gradient of vertical wind speed (nproma,nlevp1,nblks_c)


@dataclass(frozen=True)
class DiffusionMetricState:
    """Represents the metric state fields needed in diffusion."""

    theta_ref_mc: Field[[CellDim, KDim], float]
    wgtfac_c: Field[
        [CellDim, KDim], float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    mask_hdiff: Field[[CellDim, KDim], bool]
    zd_vertoffset: Field[[CECDim, KDim], int32]
    zd_diffcoef: Field[[CellDim, KDim], float]
    zd_intcoef: Field[[CECDim, KDim], float]


@dataclass(frozen=True)
class DiffusionInterpolationState:
    """Represents the ICON interpolation state needed in diffusion."""

    e_bln_c_s: Field[[CEDim], float]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: Field[[CEDim], float]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: Field[
        [CellDim, C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: Field[[CellDim, C2E2CODim], float]
    geofac_grg_y: Field[
        [CellDim, C2E2CODim], float
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: Field[[EdgeDim], float]  # Nudgeing coeffients for edges

    @functools.cached_property
    def geofac_n2s_c(self) -> Field[[CellDim], float]:
        return as_field((CellDim,), data=self.geofac_n2s.ndarray[:, 0])

    @functools.cached_property
    def geofac_n2s_nbh(self) -> Field[[CECDim], float]:
        geofac_nbh_ar = self.geofac_n2s.ndarray[:, 1:]
        old_shape = geofac_nbh_ar.shape
        return as_field(
            (CECDim,),
            geofac_nbh_ar.reshape(
                old_shape[0] * old_shape[1],
            ),
        )


@dataclass
class DiffusionOutputIntermediateFields:
    """
    For intermediate output fields
    """

    output_nabla2_diff: Field[[EdgeDim, KDim], float]
    output_nabla4_diff: Field[[EdgeDim, KDim], float]
    output_w_nabla4_diff: Field[[CellDim, KDim], float]
    
    @classmethod
    def allocate(cls, grid: BaseGrid):
        return DiffusionOutputIntermediateFields(
            output_nabla2_diff=_allocate(EdgeDim, KDim, grid=grid),
            output_nabla4_diff=_allocate(EdgeDim, KDim, grid=grid),
            output_w_nabla4_diff=_allocate(CellDim, KDim, grid=grid, is_halfdim=True),
        )


def zero_field(grid, *dims: Dimension, is_halfdim=False, dtype=float):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return as_field(dims, xp.zeros(shapex, dtype=dtype))


def _allocate(*dims: Dimension, grid, is_halfdim=False, dtype=float):
    return zero_field(grid, *dims, is_halfdim=is_halfdim, dtype=dtype)

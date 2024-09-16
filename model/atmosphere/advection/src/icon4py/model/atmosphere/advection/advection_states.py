# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@dataclasses.dataclass(frozen=True)
class AdvectionDiagnosticState:
    """Represents the diagnostic fields needed in advection."""

    #: mass of air in layer at physics time step now [kg/m^2]
    airmass_now: fa.CellKField[wpfloat]

    #: mass of air in layer at physics time step new [kg/m^2] 
    airmass_new: fa.CellKField[wpfloat]

    #: tracer tendency field for use in grid refinement [kg/kg/s] 
    grf_tend_tracer: fa.CellKField[wpfloat]

    # horizontal tracer flux at edges [kg/m/s] (nproma,nlev,nblks_e)
    hfl_tracer: fa.EdgeKField[wpfloat]

    # vertical tracer flux at cells [kg/m/s] (nproma,nlevp1,nblks_c)
    # TODO (dastrm): should be KHalfDim
    vfl_tracer: fa.CellKField[wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionInterpolationState:
    """Represents the interpolation state needed in advection."""

    #: factor for divergence 
    geofac_div: gtx.Field[[dims.CEDim], wpfloat]

    #: coefficients used for rbf interpolation of the tangential velocity component
    rbf_vec_coeff_e: gtx.Field[[dims.EdgeDim, dims.E2C2EDim], wpfloat]

    #: x-components of positions of various points on local plane tangential to the edge midpoint 
    pos_on_tplane_e_1: gtx.Field[[dims.ECDim], wpfloat]

    #: y-components of positions of various points on local plane tangential to the edge midpoint
    pos_on_tplane_e_2: gtx.Field[[dims.ECDim], wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionLeastSquaresState:
    """Represents the least squares state needed in advection."""

    #: pseudo (or Moore-Penrose) inverse of lsq design matrix A
    lsq_pseudoinv_1: gtx.Field[[dims.CECDim], wpfloat]
    lsq_pseudoinv_2: gtx.Field[[dims.CECDim], wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionMetricState:
    """Represents the metric fields needed in advection."""

    #: metrical modification factor for horizontal part of divergence at full levels (KDim)
    deepatmo_divh: fa.KField[wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (KDim)
    deepatmo_divzl: fa.KField[wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (KDim)
    deepatmo_divzu: fa.KField[wpfloat]

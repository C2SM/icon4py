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
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim


@dataclasses.dataclass(frozen=True)
class AdvectionDiagnosticState:
    """Represents the diagnostic fields needed in advection."""

    # fields for 3D elements in turbdiff
    airmass_now: fa.CellKField[
        float
    ]  # mass of air in layer at physics time step now [kg/m^2]  (nproma,nlev,nblks_c)

    airmass_new: fa.CellKField[
        float
    ]  # mass of air in layer at physics time step new [kg/m^2]  (nproma,nlev,nblks_c)

    # grf_tend_tracer: fa.CellKField[
    #    float
    # ]  # tracer tendency field for use in grid refinement [kg/kg/s]  (nproma,nlev,nblks_c)

    hfl_tracer: fa.EdgeKField[
        float
    ]  # horizontal tracer flux at edges [kg/m/s]  (nproma,nlev,nblks_e)

    # TODO (dastrm): should be KHalfDim
    vfl_tracer: fa.CellKField[
        float
    ]  # vertical tracer flux at cells [kg/m/s]  (nproma,nlevp1,nblks_c)

    # rho_incr: fa.CellKField[
    #    float
    # ]  # moist density increment for IAU [kg/m^3]  (nproma,nlev,nblks_c)

    ddt_tracer_adv: fa.CellKField[
        float
    ]  # tracer advective tendency [kg/kg/s]  (nproma,nlev,nblks_c)


@dataclasses.dataclass(frozen=True)
class AdvectionMetricState:
    """Represents the metric fields needed in advection."""

    deepatmo_divzl: fa.KField[
        float
    ]  # metrical modification factor for vertical part of divergence at full levels  (nlev)
    deepatmo_divzu: fa.KField[
        float
    ]  # metrical modification factor for vertical part of divergence at full levels  (nlev)

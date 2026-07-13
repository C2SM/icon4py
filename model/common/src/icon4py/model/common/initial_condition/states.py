# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import _common as common_utils


@dataclasses.dataclass
class DycoreInitialFields:
    """
    Fields of the dycore diagnostic state that the initial condition provides.

    They are aliases of the fields of 'DiagnosticStateNonHydro', which lives in the
    dycore package and cannot be imported here. The initial condition fills them in
    place: 'perturbed_exner_at_cells_on_model_levels' is diagnosed from the initial
    exner (compute_exner_pert in mo_nh_stepping.f90) or, when restarting, read
    together with the advective tendencies of the previous time step, which ICON
    reads from its restart file.
    """

    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat]
    normal_wind_advective_tendency: common_utils.PredictorCorrectorPair[fa.EdgeKField[ta.vpfloat]]
    vertical_wind_advective_tendency: common_utils.PredictorCorrectorPair[fa.CellKField[ta.vpfloat]]

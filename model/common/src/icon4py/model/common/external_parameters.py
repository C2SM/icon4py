# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import icon4py.model.common.external_parameters
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.test_utils import serialbox_utils as sb


@dataclasses.dataclass
class ExternalParameters:
    """Dataclass containing external parameters."""

    topo_c: fa.CellField[float]
    topo_smt_c: fa.CellField[float]


def construct_external_parameters_state(
    savepoint: sb.ExternalParametersSavepoint, num_k_lev
) -> icon4py.model.common.external_parameters.ExternalParameters:
    return icon4py.model.common.external_parameters.ExternalParameters(
        topo_c=savepoint.topo_c(),
        topo_smt_c=savepoint.topo_smt_c(),
    )

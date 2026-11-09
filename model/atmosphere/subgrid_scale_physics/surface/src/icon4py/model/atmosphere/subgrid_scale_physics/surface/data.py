# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field metadata for the SurfaceComponent input/output contract."""

from __future__ import annotations

import dataclasses
from typing import Final

from icon4py.model.common import dimension as dims
from icon4py.model.common.states import data, model, surface_flux_state


FLUX_PORTS: Final = tuple(
    field.name for field in dataclasses.fields(surface_flux_state.SurfaceFluxState)
)

INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "surface_pressure": data.DIAGNOSTIC_CF_ATTRIBUTES["surface_pressure"],
    "temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
}

# No tendencies: the surface scheme only produces diagnostics, which the physics driver binds
# to its surface state.
OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    name: dataclasses.replace(data.SURFACE_CF_ATTRIBUTES[name], dims=(dims.CellDim,))
    for name in (*FLUX_PORTS, "surface_temperature")
}

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field metadata for the TmxComponent input/output contract."""

from __future__ import annotations

import dataclasses
from typing import Final

from icon4py.model.common import dimension as dims
from icon4py.model.common.states import data, model, surface_flux_state


#: the surface fluxes tmx reads from the physics driver's surface state, in declaration order
FLUX_PORTS: Final = tuple(
    field.name for field in dataclasses.fields(surface_flux_state.SurfaceFluxState)
)

#: the state fields tmx returns a tendency for, in the order of ``TmxTendencyState``
TENDENCY_PORTS: Final = ("temperature", "qv", "qc", "qi", "u", "v", "w")

INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
    "virtual_temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["virtual_temperature"],
    "pressure": data.DIAGNOSTIC_CF_ATTRIBUTES["pressure"],
    "u": data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"],
    "v": data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"],
    "w": data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"],
    "rho": data.PROGNOSTIC_CF_ATTRIBUTES["air_density"],
    **{f"q{s}": data.COMMON_TRACER_CF_ATTRIBUTES[f"q{s}"] for s in ("v", "c", "i", "r", "s", "g")},
    # the dycore's air mass per unit area, ICON's ``mair``
    "air_mass": model.FieldMetaData(
        standard_name="atmosphere_mass_of_air_per_unit_area_in_layer",
        long_name="air mass per unit area in the layer",
        units="kg m-2",
        icon_var_name="mair",
    ),
    **{name: data.SURFACE_CF_ATTRIBUTES[name] for name in FLUX_PORTS},
}

# Dimensions are added where the field is used, keeping the shared registry dimension-agnostic.
# The PhysicsState uses them to allocate the diagnostic output buffers.
OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    **{f"tend_{name}": data.TENDENCY_CF_ATTRIBUTES[name] for name in TENDENCY_PORTS},
    "km": model.FieldMetaData(
        standard_name="turbulent_viscosity",
        long_name="turbulent viscosity, with the surface exchange coefficient at the lowest level",
        units="kg m-1 s-1",
        icon_var_name="km",
        dims=(dims.CellDim, dims.KDim),
    ),
    "kh": model.FieldMetaData(
        standard_name="turbulent_diffusivity",
        long_name="turbulent diffusivity, with the surface exchange coefficient at the lowest level",
        units="kg m-1 s-1",
        icon_var_name="kh",
        dims=(dims.CellDim, dims.KDim),
    ),
}

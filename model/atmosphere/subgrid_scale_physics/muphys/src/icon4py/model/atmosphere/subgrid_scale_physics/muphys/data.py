# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field metadata for the MuphysGranule (Component) input/output contract."""

from __future__ import annotations

from icon4py.model.common.states import data, model


_SPECIES = ("v", "c", "r", "s", "i", "g")

# muphys species key -> common tracer-registry key.
_TRACER_KEY = dict(
    v="specific_humidity",
    c="specific_cloud",
    r="specific_rain",
    s="specific_snow",
    i="specific_ice",
    g="specific_graupel",
)

# muphys precip port -> common precip-registry key.
_PRECIP_KEY = dict(
    pflx="precipitation_flux",
    pr="rainfall_flux",
    ps="snowfall_flux",
    pi="ice_fall_flux",
    pg="graupel_fall_flux",
    pre="precipitation_energy_flux",
)

INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "dz": dict(standard_name="layer_thickness", units="m"),
    "te": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
    "p": data.DIAGNOSTIC_CF_ATTRIBUTES["pressure"],
    "rho": data.PROGNOSTIC_CF_ATTRIBUTES["air_density"],
    **{f"q{s}": data.COMMON_TRACER_CF_ATTRIBUTES[_TRACER_KEY[s]] for s in _SPECIES},
}

OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "tend_temperature": data.TENDENCY_CF_ATTRIBUTES["temperature"],
    **{f"tend_q{s}": data.TENDENCY_CF_ATTRIBUTES[_TRACER_KEY[s]] for s in _SPECIES},
    **{port: data.MICROPHYSICS_PRECIP_CF_ATTRIBUTES[key] for port, key in _PRECIP_KEY.items()},
}

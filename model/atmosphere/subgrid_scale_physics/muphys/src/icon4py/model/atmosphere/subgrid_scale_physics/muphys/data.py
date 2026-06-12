# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field metadata for the MuphysGranule (Component) input/output contract.
"""

from __future__ import annotations

from icon4py.model.common.states import data, model


# muphys species keys, in the order of the muphys ``Q`` tuple.
_SPECIES = ("v", "c", "r", "s", "i", "g")

# per-species tracer identity, pulled from the common registry.
_TRACER: dict[str, model.FieldMetaData] = {
    "v": data.COMMON_TRACER_CF_ATTRIBUTES["specific_humidity"],
    "c": data.COMMON_TRACER_CF_ATTRIBUTES["specific_cloud"],
    "r": data.COMMON_TRACER_CF_ATTRIBUTES["specific_rain"],
    "s": data.COMMON_TRACER_CF_ATTRIBUTES["specific_snow"],
    "i": data.COMMON_TRACER_CF_ATTRIBUTES["specific_ice"],
    "g": data.COMMON_TRACER_CF_ATTRIBUTES["specific_graupel"],
}


def _tendency_of(base: model.FieldMetaData) -> model.FieldMetaData:
    """Derive a tendency's metadata from the field it is the tendency of."""
    return dict(
        standard_name=f"tendency_of_{base['standard_name']}",
        units=f"{base['units']} s-1",
        kind="tendency",
    )

INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "dz": dict(standard_name="layer_thickness", units="m"),
    "te": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
    "p": data.DIAGNOSTIC_CF_ATTRIBUTES["pressure"],
    "rho": data.PROGNOSTIC_CF_ATTRIBUTES["air_density"],
    **{f"q{s}": _TRACER[s] for s in _SPECIES},
}

# muphys precip diagnostics -- process-specific (not common fields): pflx is the 3D
# flux, pr/ps/pi/pg surface rates, pre the surface precip energy flux.
_PRECIP_PROPERTIES: dict[str, model.FieldMetaData] = {
    "pflx": dict(standard_name="precipitation_flux", units="kg m-2 s-1", kind="diagnostic"),
    "pr": dict(standard_name="rainfall_flux", units="kg m-2 s-1", kind="diagnostic"),
    "ps": dict(standard_name="snowfall_flux", units="kg m-2 s-1", kind="diagnostic"),
    "pi": dict(standard_name="ice_fall_flux", units="kg m-2 s-1", kind="diagnostic"),
    "pg": dict(standard_name="graupel_fall_flux", units="kg m-2 s-1", kind="diagnostic"),
    "pre": dict(standard_name="precipitation_energy_flux", units="W m-2", kind="diagnostic"),
}

# muphys outputs: 7 tendencies (derived from the input identities) + 6 precip diagnostics
OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "tend_temperature": _tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"]),
    **{f"tend_q{s}": _tendency_of(_TRACER[s]) for s in _SPECIES},
    **_PRECIP_PROPERTIES,
}

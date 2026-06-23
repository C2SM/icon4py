# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

from icon4py.model.common.states import model


"""Static metadata for common fields in the model."""

#: CF attributes of the prognostic variables
PROGNOSTIC_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    air_density=dict(
        standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"
    ),
    virtual_potential_temperature=dict(
        standard_name="virtual_potential_temperature",
        long_name="virtual potential temperature",
        units="K",
        icon_var_name="theta_v",
    ),
    exner_function=dict(
        standard_name="dimensionless_exner_function",
        long_name="exner function",
        icon_var_name="exner",
        units="1",
    ),
    upward_air_velocity=dict(
        standard_name="upward_air_velocity",
        long_name="vertical air velocity component",
        units="m s-1",
        icon_var_name="w",
        is_on_half_levels=True,
    ),
    normal_velocity=dict(
        standard_name="normal_velocity",
        long_name="velocity normal to edge",
        units="m s-1",
        icon_var_name="vn",
    ),
    tangential_velocity=dict(
        standard_name="tangential_velocity",
        long_name="velocity tangential to edge",
        units="m s-1",
        icon_var_name="vt",
    ),
)

#: CF attributes of common tracer variables
COMMON_TRACER_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    specific_humidity=dict(
        standard_name="specific_humidity",
        long_name="ratio of water vapor mass to total moist air parcel mass",
        units="1",
        icon_var_name="qv",
    ),
    specific_cloud=dict(
        standard_name="specific_cloud_content",
        long_name="ratio of cloud water mass to total moist air parcel mass",
        units="1",
        icon_var_name="qc",
    ),
    specific_ice=dict(
        standard_name="specific_ice_content",
        long_name="ratio of cloud ice mass to total moist air parcel mass",
        units="1",
        icon_var_name="qi",
    ),
    specific_rain=dict(
        standard_name="specific_rain_content",
        long_name="ratio of rain mass to total moist air parcel mass",
        units="1",
        icon_var_name="qr",
    ),
    specific_snow=dict(
        standard_name="specific_snow_content",
        long_name="ratio of snow mass to total moist air parcel mass",
        units="1",
        icon_var_name="qs",
    ),
    specific_graupel=dict(
        standard_name="specific_graupel_content",
        long_name="ratio of graupel mass to total moist air parcel mass",
        units="1",
        icon_var_name="qg",
    ),
)

#: CF attributes of diagnostic variables
DIAGNOSTIC_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    eastward_wind=dict(
        standard_name="eastward_wind",
        long_name="eastward wind component",
        units="m s-1",
        icon_var_name="u",
    ),
    northward_wind=dict(
        standard_name="northward_wind",
        long_name="northward wind component",
        units="m s-1",
        icon_var_name="v",
    ),
    temperature=dict(
        standard_name="air_temperature",
        long_name="air temperature",
        units="K",
        icon_var_name="temp",
    ),
    virtual_temperature=dict(
        standard_name="air_virtual_temperature",
        long_name="air virtual temperature",
        units="K",
        icon_var_name="tempv",
    ),
    pressure=dict(
        standard_name="air_pressure",
        long_name="air pressure",
        units="Pa",
        icon_var_name="pres",
    ),
    surface_pressure=dict(
        standard_name="air_pressure_at_ground_level",
        long_name="air pressure at ground level",
        units="Pa",
        icon_var_name="pres_sfc",
    ),
)

# CF attributes of microphysics precipitation-flux diagnostics.
# Shared across microphysics schemes (muphys, NWP graupel)
MICROPHYSICS_PRECIP_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    precipitation_flux=dict(
        standard_name="precipitation_flux",
        long_name="precipitation flux",
        units="kg m-2 s-1",
        kind="diagnostic",
    ),
    rainfall_flux=dict(
        standard_name="rainfall_flux",
        long_name="rainfall flux",
        units="kg m-2 s-1",
        kind="diagnostic",
    ),
    snowfall_flux=dict(
        standard_name="snowfall_flux",
        long_name="snowfall flux",
        units="kg m-2 s-1",
        kind="diagnostic",
    ),
    graupel_fall_flux=dict(
        standard_name="graupel_fall_flux",
        long_name="graupel fall flux",
        units="kg m-2 s-1",
        kind="diagnostic",
    ),
    ice_fall_flux=dict(
        standard_name="ice_fall_flux",
        long_name="ice fall flux",
        units="kg m-2 s-1",
        kind="diagnostic",
    ),
    precipitation_energy_flux=dict(
        standard_name="precipitation_energy_flux",
        long_name="precipitation energy flux",
        units="W m-2",
        kind="diagnostic",
    ),
)


def tendency_of(base: model.FieldMetaData) -> model.FieldMetaData:
    """Derive generic tendency metadata for ``base`` (CF ``tendency_of_<name>``).

    The tendency carries ``kind="tendency"`` and ``base``'s units per second; a
    dimensionless base (``units="1"``) yields ``"s-1"`` rather than ``"1 s-1"``.
    """
    base_units = base["units"]
    units = "s-1" if base_units == "1" else f"{base_units} s-1"
    tendency: model.FieldMetaData = dict(
        standard_name=f"tendency_of_{base['standard_name']}",
        units=units,
        kind="tendency",
    )
    long_name = base.get("long_name")
    if long_name is not None:
        tendency["long_name"] = f"tendency of {long_name}"
    return tendency


# Generic tendencies of the temperature and tracer fields, derived from their base identities so names/units never drift.
TENDENCY_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    temperature=tendency_of(DIAGNOSTIC_CF_ATTRIBUTES["temperature"]),
    specific_humidity=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["specific_humidity"]),
    specific_cloud=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["specific_cloud"]),
    specific_rain=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["specific_rain"]),
    specific_snow=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["specific_snow"]),
    specific_ice=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["specific_ice"]),
    specific_graupel=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["specific_graupel"]),
)

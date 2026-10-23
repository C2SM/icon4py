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
    air_density=model.FieldMetaData(
        standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"
    ),
    virtual_potential_temperature=model.FieldMetaData(
        standard_name="virtual_potential_temperature",
        long_name="virtual potential temperature",
        units="K",
        icon_var_name="theta_v",
    ),
    exner_function=model.FieldMetaData(
        standard_name="dimensionless_exner_function",
        long_name="exner function",
        icon_var_name="exner",
        units="1",
    ),
    upward_air_velocity=model.FieldMetaData(
        standard_name="upward_air_velocity",
        long_name="vertical air velocity component",
        units="m s-1",
        icon_var_name="w",
    ),
    normal_velocity=model.FieldMetaData(
        standard_name="normal_velocity",
        long_name="velocity normal to edge",
        units="m s-1",
        icon_var_name="vn",
    ),
    tangential_velocity=model.FieldMetaData(
        standard_name="tangential_velocity",
        long_name="velocity tangential to edge",
        units="m s-1",
        icon_var_name="vt",
    ),
)

#: index of tracer variables in the serialized fortran list
QV: Final[int] = 0
QC: Final[int] = 1
QI: Final[int] = 2
QR: Final[int] = 3
QS: Final[int] = 4
QG: Final[int] = 5

#: CF attributes of common tracer variables
COMMON_TRACER_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    qv=model.FieldMetaData(
        standard_name="specific_humidity",
        long_name="ratio of water vapor mass to total moist air parcel mass",
        units="1",
        icon_var_name="qv",
        icon_var_list_index=QV,
    ),
    qc=model.FieldMetaData(
        standard_name="specific_cloud_content",
        long_name="ratio of cloud water mass to total moist air parcel mass",
        units="1",
        icon_var_name="qc",
        icon_var_list_index=QC,
    ),
    qi=model.FieldMetaData(
        standard_name="specific_ice_content",
        long_name="ratio of cloud ice mass to total moist air parcel mass",
        units="1",
        icon_var_name="qi",
        icon_var_list_index=QI,
    ),
    qr=model.FieldMetaData(
        standard_name="specific_rain_content",
        long_name="ratio of rain mass to total moist air parcel mass",
        units="1",
        icon_var_name="qr",
        icon_var_list_index=QR,
    ),
    qs=model.FieldMetaData(
        standard_name="specific_snow_content",
        long_name="ratio of snow mass to total moist air parcel mass",
        units="1",
        icon_var_name="qs",
        icon_var_list_index=QS,
    ),
    qg=model.FieldMetaData(
        standard_name="specific_graupel_content",
        long_name="ratio of graupel mass to total moist air parcel mass",
        units="1",
        icon_var_name="qg",
        icon_var_list_index=QG,
    ),
)

#: CF attributes of diagnostic variables
DIAGNOSTIC_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    eastward_wind=model.FieldMetaData(
        standard_name="eastward_wind",
        long_name="eastward wind component",
        units="m s-1",
        icon_var_name="u",
    ),
    northward_wind=model.FieldMetaData(
        standard_name="northward_wind",
        long_name="northward wind component",
        units="m s-1",
        icon_var_name="v",
    ),
    temperature=model.FieldMetaData(
        standard_name="air_temperature",
        long_name="air temperature",
        units="K",
        icon_var_name="temp",
    ),
    virtual_temperature=model.FieldMetaData(
        standard_name="air_virtual_temperature",
        long_name="air virtual temperature",
        units="K",
        icon_var_name="tempv",
    ),
    pressure=model.FieldMetaData(
        standard_name="air_pressure",
        long_name="air pressure",
        units="Pa",
        icon_var_name="pres",
    ),
    surface_pressure=model.FieldMetaData(
        standard_name="air_pressure_at_ground_level",
        long_name="air pressure at ground level",
        units="Pa",
        icon_var_name="pres_sfc",
    ),
)

# CF attributes of precipitation-flux diagnostics.
PRECIPITATION_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    precipitation_flux=model.FieldMetaData(
        standard_name="precipitation_flux",
        long_name="total precipitation flux (rain + snow + graupel + ice)",
        units="kg m-2 s-1",
    ),
    rainfall_flux=model.FieldMetaData(
        standard_name="rainfall_flux",
        long_name="rainfall flux",
        units="kg m-2 s-1",
    ),
    snowfall_flux=model.FieldMetaData(
        standard_name="snowfall_flux",
        long_name="snowfall flux",
        units="kg m-2 s-1",
    ),
    graupelfall_flux=model.FieldMetaData(
        standard_name="graupelfall_flux",
        long_name="graupelfall flux",
        units="kg m-2 s-1",
    ),
    icefall_flux=model.FieldMetaData(
        standard_name="icefall_flux",
        long_name="icefall flux",
        units="kg m-2 s-1",
    ),
    precipitation_energy_flux=model.FieldMetaData(
        standard_name="precipitation_energy_flux",
        long_name="energy flux carried by precipitation",
        units="W m-2",
    ),
)


def tendency_of(base: model.FieldMetaData) -> model.FieldMetaData:
    """Derive generic tendency metadata for ``base`` (CF ``tendency_of_<name>``).

    The tendency carries ``kind=FieldKind.TENDENCY`` and ``base``'s units per
    second; a dimensionless base (``units="1"``) yields ``"s-1"`` rather than
    ``"1 s-1"``.
    """
    units = "s-1" if base.units == "1" else f"{base.units} s-1"
    return model.FieldMetaData(
        standard_name=f"tendency_of_{base.standard_name}",
        units=units,
        kind=model.FieldKind.TENDENCY,
        long_name=None if base.long_name is None else f"tendency of {base.long_name}",
    )


# Generic tendencies of the temperature and tracer fields, derived from their base identities so names/units never drift.
TENDENCY_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    temperature=tendency_of(DIAGNOSTIC_CF_ATTRIBUTES["temperature"]),
    qv=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["qv"]),
    qc=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["qc"]),
    qi=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["qi"]),
    qr=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["qr"]),
    qs=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["qs"]),
    qg=tendency_of(COMMON_TRACER_CF_ATTRIBUTES["qg"]),
    u=tendency_of(DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"]),
    v=tendency_of(DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"]),
    w=tendency_of(PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"]),
)

#: CF attributes of the surface fields the surface scheme hands to the atmosphere, keyed by
#: the member names of ``SurfaceFluxState`` and of the physics driver's surface state.
SURFACE_CF_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    evapotranspiration=model.FieldMetaData(
        standard_name="water_evapotranspiration_flux",
        long_name="surface evapotranspiration flux",
        units="kg m-2 s-1",
        icon_var_name="evspsbl",
    ),
    sensible_heat_flux=model.FieldMetaData(
        standard_name="surface_upward_sensible_heat_flux",
        long_name="surface sensible heat flux",
        units="W m-2",
        icon_var_name="hfss",
    ),
    latent_heat_flux=model.FieldMetaData(
        standard_name="surface_upward_latent_heat_flux",
        long_name="surface latent heat flux",
        units="W m-2",
        icon_var_name="hfls",
    ),
    u_stress=model.FieldMetaData(
        standard_name="surface_downward_eastward_stress",
        long_name="zonal surface wind stress",
        units="N m-2",
        icon_var_name="tauu",
    ),
    v_stress=model.FieldMetaData(
        standard_name="surface_downward_northward_stress",
        long_name="meridional surface wind stress",
        units="N m-2",
        icon_var_name="tauv",
    ),
    q_snocpymlt=model.FieldMetaData(
        standard_name="surface_snow_and_canopy_melt_heat_flux",
        long_name="heat used to melt snow on the canopy",
        units="W m-2",
        icon_var_name="q_snocpymlt",
    ),
    thermal_energy_flux=model.FieldMetaData(
        standard_name="surface_upward_internal_energy_flux_due_to_thermal_exchange",
        long_name="surface flux of internal energy from thermal exchange",
        units="W m-2",
        icon_var_name="ufts",
    ),
    vapor_energy_flux=model.FieldMetaData(
        standard_name="surface_upward_internal_energy_flux_due_to_vapor_exchange",
        long_name="surface flux of internal energy from vapor exchange",
        units="W m-2",
        icon_var_name="ufvs",
    ),
    surface_temperature=model.FieldMetaData(
        standard_name="surface_temperature",
        long_name="surface temperature",
        units="K",
        icon_var_name="ts",
    ),
    exchange_coefficient_momentum=model.FieldMetaData(
        standard_name="surface_exchange_coefficient_for_momentum",
        long_name="surface exchange coefficient for momentum",
        units="1",
        icon_var_name="km_sfc",
    ),
    exchange_coefficient_heat=model.FieldMetaData(
        standard_name="surface_exchange_coefficient_for_heat",
        long_name="surface exchange coefficient for heat",
        units="1",
        icon_var_name="kh_sfc",
    ),
)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field metadata for the TmxComponent input/output contract."""

from __future__ import annotations

from icon4py.model.common.states import data, model


_TMX_ONLY_INPUTS: dict[str, model.FieldMetaData] = {
    "pressure_ifc": dict(standard_name="air_pressure_on_interface_levels", units="Pa"),
    "air_mass": dict(standard_name="air_mass_per_unit_area", units="kg m-2"),
    "cv_air": dict(
        standard_name="isometric_heat_capacity_of_moist_air_per_unit_area", units="J m-2 K-1"
    ),
    "evapotranspiration": dict(standard_name="surface_evapotranspiration_flux", units="kg m-2 s-1"),
    "sensible_heat_flux": dict(standard_name="surface_upward_sensible_heat_flux", units="W m-2"),
    "u_stress": dict(standard_name="surface_downward_eastward_stress", units="N m-2"),
    "v_stress": dict(standard_name="surface_downward_northward_stress", units="N m-2"),
    "q_snocpymlt": dict(standard_name="heating_used_to_melt_snow_on_canopy", units="W m-2"),
}

INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
    "virtual_temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["virtual_temperature"],
    "pressure": data.DIAGNOSTIC_CF_ATTRIBUTES["pressure"],
    "u": data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"],
    "v": data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"],
    "w": data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"],
    "rho": data.PROGNOSTIC_CF_ATTRIBUTES["air_density"],
    **{f"q{s}": data.COMMON_TRACER_CF_ATTRIBUTES[f"q{s}"] for s in "vcirsg"},
    **_TMX_ONLY_INPUTS,
}

OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "ddt_temperature": data.TENDENCY_CF_ATTRIBUTES["temperature"],
    "ddt_qv": data.TENDENCY_CF_ATTRIBUTES["qv"],
    "ddt_qc": data.TENDENCY_CF_ATTRIBUTES["qc"],
    "ddt_qi": data.TENDENCY_CF_ATTRIBUTES["qi"],
    "ddt_u": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"]),
    "ddt_v": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"]),
    "ddt_w": data.tendency_of(data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"]),
    "km": dict(standard_name="mass_weighted_turbulent_viscosity", units="kg m-1 s-1"),
    "kh": dict(standard_name="mass_weighted_turbulent_diffusivity", units="kg m-1 s-1"),
    "heating": dict(standard_name="turbulent_heating_rate", units="W m-2"),
    "dissip_ke": dict(standard_name="kinetic_energy_dissipation_rate", units="W m-2"),
    "cptgz_vi": dict(standard_name="vertically_integrated_dry_static_energy", units="J m-2"),
    "dissip_ke_vi": dict(
        standard_name="vertically_integrated_kinetic_energy_dissipation_rate", units="W m-2"
    ),
    "int_energy_vi": dict(standard_name="vertically_integrated_internal_energy", units="J m-2"),
    "int_energy_vi_tend": dict(
        standard_name="tendency_of_vertically_integrated_internal_energy", units="W m-2"
    ),
}

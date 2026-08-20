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


INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
    "virtual_temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["virtual_temperature"],
    "pressure": data.DIAGNOSTIC_CF_ATTRIBUTES["pressure"],
    "u": data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"],
    "v": data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"],
    "w": data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"],
    "rho": data.PROGNOSTIC_CF_ATTRIBUTES["air_density"],
    **{f"q{s}": data.COMMON_TRACER_CF_ATTRIBUTES[f"q{s}"] for s in "vcirsg"},
    "evapotranspiration": data.SURFACE_FLUX_CF_ATTRIBUTES["surface_evapotranspiration_flux"],
    "sensible_heat_flux": data.SURFACE_FLUX_CF_ATTRIBUTES["surface_upward_sensible_heat_flux"],
    "u_stress": data.SURFACE_FLUX_CF_ATTRIBUTES["surface_downward_eastward_stress"],
    "v_stress": data.SURFACE_FLUX_CF_ATTRIBUTES["surface_downward_northward_stress"],
    # TMX-interface quirks, declared locally (single consumer)
    "pressure_ifc": dict(standard_name="air_pressure_on_interface_levels", units="Pa"),
    "air_mass": dict(standard_name="air_mass_per_unit_area", units="kg m-2"),
    "cv_air": dict(
        standard_name="isometric_heat_capacity_of_moist_air_per_unit_area", units="J m-2 K-1"
    ),
    "q_snocpymlt": dict(standard_name="heating_used_to_melt_snow_on_canopy", units="W m-2"),
}

OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "tend_temperature": data.TENDENCY_CF_ATTRIBUTES["temperature"],
    "tend_qv": data.TENDENCY_CF_ATTRIBUTES["qv"],
    "tend_qc": data.TENDENCY_CF_ATTRIBUTES["qc"],
    "tend_qi": data.TENDENCY_CF_ATTRIBUTES["qi"],
    "tend_u": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"]),
    "tend_v": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"]),
    "tend_w": data.tendency_of(data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"]),
    "km": dict(
        standard_name="mass_weighted_turbulent_viscosity", kind="diagnostic", units="kg m-1 s-1"
    ),
    "kh": dict(
        standard_name="mass_weighted_turbulent_diffusivity", kind="diagnostic", units="kg m-1 s-1"
    ),
    "heating": dict(standard_name="turbulent_heating_rate", kind="diagnostic", units="W m-2"),
    "dissip_ke": dict(
        standard_name="kinetic_energy_dissipation_rate", kind="diagnostic", units="W m-2"
    ),
    "cptgz_vi": dict(
        standard_name="vertically_integrated_dry_static_energy", kind="diagnostic", units="J m-2"
    ),
    "dissip_ke_vi": dict(
        standard_name="vertically_integrated_kinetic_energy_dissipation_rate",
        kind="diagnostic",
        units="W m-2",
    ),
    "int_energy_vi": dict(
        standard_name="vertically_integrated_internal_energy", kind="diagnostic", units="J m-2"
    ),
    "int_energy_vi_tend": dict(
        standard_name="tendency_of_vertically_integrated_internal_energy",
        kind="diagnostic",
        units="W m-2",
    ),
}


# The wrapper contract speaks the AES ``tend%`` naming (shared with muphys); the
# granule's ``TmxTendencyState`` keeps upstream's ``ddt_*`` port names. This map is
# the component adapter's contract-key -> granule-field translation.
# TODO (Yilu): later on we can also rename inside the granule
TENDENCY_GRANULE_PORTS: dict[str, str] = {
    f"tend_{name}": f"ddt_{name}" for name in ("temperature", "qv", "qc", "qi", "u", "v", "w")
}

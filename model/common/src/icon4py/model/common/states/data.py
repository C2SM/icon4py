# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

import icon4py.model.common.states.model as model


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

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Annotated


GAS_CONSTANT_DRY_AIR: Annotated[
    float,
    "gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90), https://glossary.ametsoc.org/wiki/Gas_constant",
] = 287.04
CPD: Annotated[float, "specific heat at constant pressure [J/K/kg]"] = 1004.64
GAS_CONSTANT_WATER_VAPOR: Annotated[
    float, "gas constant for water vapor [J/K/kg], rv in Icon"
] = 461.51
GRAVITATIONAL_ACCELERATION: Annotated[
    float, "av. gravitational acceleration [m/s^2]"
] = 9.8066
DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO: Annotated[
    float, "default physics to dynamics time step ratio"
] = 5.0

# nonhydro constants
RD: Annotated[float, "[J/K/kg] gas constant"] = 287.04
CVD: Annotated[float, "[J/K/kg] specific heat at constant volume"] = CPD - RD
RAYLEIGH_KLEMP: Annotated[int, "Klemp (2008) type Rayleigh damping"] = 2
P0REF: Annotated[float, "[Pa]  reference pressure for Exner function"] = 100000.0
CVD_O_RD = CVD / RD

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

from typing import Final


#: Gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90),
#: see https://glossary.ametsoc.org/wiki/Gas_constant.
GAS_CONSTANT_DRY_AIR: Final[float] = 287.04

#: Specific heat at constant pressure [J/K/kg]
CPD: Final[float] = 1004.64

#: Gas constant for water vapor [J/K/kg], rv in ICON.
GAS_CONSTANT_WATER_VAPOR: Final[float] = 461.51

#: Av. gravitational acceleration [m/s^2]
GRAVITATIONAL_ACCELERATION: Final[float] = 9.8066

#: Default physics to dynamics time step ratio
DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO: Final[float] = 5.0

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

from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import (
    diagnose_pressure as diagnose_pressure_orig,
)
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure as diagnose_surface_pressure_orig,
)
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_temperature import (
    diagnose_temperature as diagnose_temperature_orig,
)
from icon4py.model.common.caching import CachedProgram
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation as edge_2_cell_vector_rbf_interpolation_orig,
)

# diagnostic stencils
diagnose_pressure = CachedProgram(diagnose_pressure_orig)
diagnose_surface_pressure = CachedProgram(diagnose_surface_pressure_orig)
diagnose_temperature = CachedProgram(diagnose_temperature_orig)
edge_2_cell_vector_rbf_interpolation = CachedProgram(
    edge_2_cell_vector_rbf_interpolation_orig
)

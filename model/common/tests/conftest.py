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


from icon4py.model.common.test_utils.fixtures import (  # noqa F401
    damping_height,
    data_provider,
    datapath,
    get_grid_files,
    grid_savepoint,
    icon_grid,
    r04b09_dsl_gridfile,
    setup_icon_data,


)
from icon4py.model.common.test_utils.parallel_fixtures import (processor_props, ranked_data_path, get_decomposition_info, get_icon_grid, download_data)  # noqa F401

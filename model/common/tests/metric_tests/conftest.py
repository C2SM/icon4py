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
import pytest
from gt4py.next.program_processors.otf_compile_executor import OTFCompileExecutor

from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
)
from icon4py.model.common.test_utils.helpers import (  # noqa : F401  # fixtures from test_utils
    backend,
)


@pytest.fixture
def is_otf(backend) -> bool:  # noqa : F811 # fixture is used in the test
    # not reusing the `uses_icon_grid_with_otf` fixture because it also checks for the grid
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    if hasattr(backend, "executor"):
        if isinstance(backend.executor, OTFCompileExecutor):
            return True
    return False

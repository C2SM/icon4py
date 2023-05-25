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

from pathlib import Path

import pytest

from . import test_utils


@pytest.fixture
def samples_path():
    return Path(test_utils.__file__).parent


@pytest.fixture
def diffusion_granule(samples_path):
    return samples_path / "diffusion_granule.f90"


@pytest.fixture
def diffusion_granule_deps(samples_path):
    return [samples_path / "derived_types_example.f90"]


@pytest.fixture
def no_deps_source_file(samples_path):
    return samples_path / "no_deps_subroutine_example.f90"


@pytest.fixture
def not_existing_diffusion_granule(samples_path):
    return samples_path / "not_existing_file.f90"

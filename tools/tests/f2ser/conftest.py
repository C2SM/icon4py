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

from importlib.resources import files

import pytest


@pytest.fixture
def samples_path():
    return files("icon4pytools.data")


@pytest.fixture
def diffusion_granule(samples_path):
    return samples_path.joinpath("diffusion_granule.f90")


@pytest.fixture
def diffusion_granule_deps(samples_path):
    return [samples_path.joinpath("derived_types_example.f90")]


@pytest.fixture
def no_deps_source_file(samples_path):
    return samples_path.joinpath("no_deps_subroutine_example.f90")


@pytest.fixture
def not_existing_diffusion_granule(samples_path):
    return samples_path.joinpath("not_existing_file.f90")

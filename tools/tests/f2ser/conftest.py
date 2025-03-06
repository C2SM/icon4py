# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "fortran_samples"


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

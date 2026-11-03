# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest


def test_importing_the_model_does_not_let_mpi4py_initialize_mpi():
    """A netCDF4 built with parallel support imports mpi4py when it is loaded, and mpi4py
    initializes MPI on import by default. icon4py initializes MPI itself
    (``decomposition.mpi_decomposition.init_mpi``), so loading the model, in any import
    order, must leave that to it: ``icon4py.model.common`` turns the automatic
    initialization off before anything else of the model is imported."""
    mpi4py = pytest.importorskip("mpi4py")
    importlib.import_module("icon4py.model.common")
    assert mpi4py.rc.initialize is False
    importlib.import_module("icon4py.model.common.io.netcdf_writers")
    assert mpi4py.rc.initialize is False

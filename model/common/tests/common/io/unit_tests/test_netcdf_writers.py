# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest


def test_importing_the_model_does_not_let_mpi4py_initialize_mpi() -> None:
    """Importing the model leaves ``mpi4py.rc.initialize`` off, in any import order.

    A parallel netCDF4 build imports mpi4py, which initializes MPI on import by default;
    icon4py initializes MPI itself (see "Parallel netCDF" in the ``io`` docs).
    """
    mpi4py = pytest.importorskip("mpi4py")
    importlib.import_module("icon4py.model.common")
    assert mpi4py.rc.initialize is False
    importlib.import_module("icon4py.model.common.io.netcdf_writers")
    assert mpi4py.rc.initialize is False

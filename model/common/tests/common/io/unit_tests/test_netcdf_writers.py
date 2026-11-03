# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest


def test_importing_the_netcdf_writer_does_not_let_mpi4py_initialize_mpi():
    """A netCDF4 built with parallel support imports mpi4py at import time, and mpi4py
    initializes MPI on import by default. icon4py initializes MPI itself
    (``decomposition.mpi_decomposition.init_mpi``), so importing the writer must leave
    that to it -- also inside GT4Py's spawned compile workers, which re-import the
    model's modules."""
    mpi4py = pytest.importorskip("mpi4py")
    importlib.import_module("icon4py.model.common.io.netcdf_writers")
    assert mpi4py.rc.initialize is False

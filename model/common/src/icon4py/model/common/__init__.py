# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

from packaging import version as pkg_version


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
]


__author__: Final = "ETH Zurich,  MeteoSwiss and individual contributors"
__copyright__: Final = "Copyright (c) 2022-2024 ETH Zurich and MeteoSwiss"
__license__: Final = "BSD-3-Clause"


__version__: Final = "0.5.0"
__version_info__: Final = pkg_version.parse(__version__)


def _disable_mpi4py_auto_initialization() -> None:
    """Leave the initialization of MPI to icon4py, whatever gets imported first.

    icon4py initializes MPI itself (``decomposition.mpi_decomposition.init_mpi``,
    funneled). mpi4py's default is to initialize MPI when ``mpi4py.MPI`` is first
    imported, at the highest thread level -- and that import happens behind the
    model's back as soon as a netCDF4 built with parallel support is loaded, which
    the model's own import graph does before ``mpi_decomposition`` runs. The flag is
    turned off here, before any other module of the model loads, so that it cannot
    happen in any import order: also in GT4Py's spawned compile workers, which
    re-import the model's modules and would otherwise join the real ranks' PMI
    bootstrap and break it.
    """
    try:
        import mpi4py  # noqa: PLC0415 [import-outside-top-level]
    except ImportError:  # serial installation
        return
    mpi4py.rc.initialize = False


_disable_mpi4py_auto_initialization()

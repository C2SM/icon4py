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
    """Leave MPI initialization to icon4py; see "Parallel netCDF" in the ``io`` docs."""
    try:
        import mpi4py  # noqa: PLC0415 [import-outside-top-level]
    except ImportError:  # serial installation
        return
    mpi4py.rc.initialize = False


_disable_mpi4py_auto_initialization()

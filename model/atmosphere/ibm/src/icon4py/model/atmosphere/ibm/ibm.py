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
import functools
import logging
import math
import sys
import dataclasses
import enum
from typing import Final, Optional

from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import CellDim, KDim


"""
Immersed boundary method module

"""


class ImmersedBoundaryMethod:
    """
    Main class for the immersed boundary method.
    """

    def __init__(
        self,
        log: logging.Logger = None,
    ):
        self.log = log or logging.getLogger(__name__)

        self._validate_config()

        self.mask: Field[[CellDim, KDim], int] = 0

        self.log.info("IBM initialized")

    def _validate_config(self):
        pass

    def set_boundary_conditions(self):
        self.log.info("IBM set BCs ")
        pass
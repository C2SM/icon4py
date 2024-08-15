# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import enum

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid

"""
Advection module ported from ICON mo_advection_stepping.f90.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class HorizontalAdvectionType(int, enum.Enum):
    """
    Horizontal operator scheme for advection.
    """

    NO_ADVECTION = 0  #: no horizontal advection
    LINEAR_2ND_ORDER = 2  #: 2nd order MIURA with linear reconstruction


class HorizontalAdvectionLimiterType(int, enum.Enum):
    """
    Limiter for horizontal advection operator.
    """

    NO_LIMITER = 0  #: no horizontal limiter
    POSITIVE_DEFINITE = 4  #: positive definite horizontal limiter


class AdvectionConfig:
    """
    Contains necessary parameters to configure an advection run.

    Default values match a basic implementation.
    """

    def __init__(
        self,
        horizontal_advection_type: HorizontalAdvectionType = HorizontalAdvectionType.LINEAR_2ND_ORDER,
        horizontal_advection_limiter: HorizontalAdvectionLimiterType = HorizontalAdvectionLimiterType.POSITIVE_DEFINITE,
    ):
        """Set the default values according to a basic implementation."""

        self.horizontal_advection_type: int = horizontal_advection_type
        self.horizontal_advection_limiter: int = horizontal_advection_limiter

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.horizontal_advection_type != 2:
            raise NotImplementedError(
                "Only horizontal advection type 2 = `2nd order MIURA with linear reconstruction` is implemented"
            )
        if self.horizontal_advection_limiter != 4:
            raise NotImplementedError(
                "Only horizontal advection limiter 4 = `positive definite limiter` is implemented"
            )


class Advection:
    """Class that configures advection and does one advection step."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        config: AdvectionConfig,
        # params: AdvectionParams,
        edge_params: h_grid.EdgeParams,
        cell_params: h_grid.CellParams,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        """
        Initialize advection granule with configuration.
        """
        log.debug("advection class init - start")

        self.config: AdvectionConfig = config
        # self.params: AdvectionParams = params
        self.grid = grid
        self.edge_params = edge_params
        self.cell_params = cell_params
        self._exchange = exchange

        log.debug("advection class init - done")

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum

from icon4py.model.common.config import config_io


@config_io.register_enum
class HorizontalAdvectionType(Enum):
    """
    Horizontal operator scheme for tracer advection (originally ihadv_tracer).
    """

    #: no horizontal tracer advection
    NO_ADVECTION = 0
    #: 1st order upwind
    FIRST_ORDER_UPWIND = 1
    #: 2nd order MIURA with linear reconstruction
    SECOND_ORDER_LINEAR_MIURA = 2


@config_io.register_enum
class HorizontalAdvectionLimiter(Enum):
    """
    Limiter for horizontal tracer advection operator (originally itype_hlimit).
    """

    #: no horizontal limiter
    NO_LIMITER = 0
    #: positive definite horizontal limiter
    POSITIVE_DEFINITE = 4


@config_io.register_enum
class VerticalAdvectionType(Enum):
    """
    Vertical operator scheme for tracer advection (originally ivadv_tracer).
    """

    #: no vertical tracer advection
    NO_ADVECTION = 0
    #: 1st order upwind
    FIRST_ORDER_UPWIND = 1
    #: 3rd order PPM
    THIRD_ORDER_PPM = 3


@config_io.register_enum
class VerticalAdvectionLimiter(Enum):
    """
    Limiter for vertical tracer advection operator (originally itype_vlimit).
    """

    #: no vertical limiter
    NO_LIMITER = 0
    #: semi-monotonic vertical limiter
    SEMI_MONOTONIC = 1

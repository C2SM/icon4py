# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Time-related type aliases shared across the model."""

import datetime
from typing import TypeAlias


RelativeTime: TypeAlias = datetime.timedelta
AbsoluteTime: TypeAlias = datetime.datetime
NumTimeSteps: TypeAlias = int
EndOfSimulation: TypeAlias = RelativeTime | AbsoluteTime | NumTimeSteps

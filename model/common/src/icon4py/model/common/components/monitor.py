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

import abc
from abc import ABC
from datetime import datetime


class Monitor(ABC):
    """
    Monitor component of the model.

    Monitor is a base class for components that store or freeze state for later usage but
    don't modify it or return any new state objects.

    Named after Sympl Monitor component: https://sympl.readthedocs.io/en/latest/monitors.html
    """

    def __str__(self):
        return "instance of {}(Monitor)".format(self.__class__)

    # TODO (halungge) rename __call__ and make it callable, as the other components?
    @abc.abstractmethod
    def store(self, state: dict, model_time: datetime, *args, **kwargs) -> None:
        """Store state and perform class specific actions on it.


        Args:
            state: dict  model state dictionary
            model_time: current simulation time
        """
        pass

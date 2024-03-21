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
from dataclasses import dataclass
from datetime import datetime, timedelta


def to_delta(value: str) -> timedelta:
    vals = value.split(" ")
    num = 1 if not vals[0].isnumeric() else int(vals[0]) 

    value = vals[0].upper() if len(vals) < 2 else vals[1].upper()
    if value== "HOUR":
        return timedelta(hours=num)
    elif value == "DAY":
        return timedelta(days=num)
    elif value == "MINUTE":
        return timedelta(minutes=num)
    elif value == "SECOND":
        return timedelta(seconds=num)
    else:
        raise NotImplementedError(f" delta {value} is not supported")


class Config(ABC):
    """
    Base class for all config classes.
    """

    def __str__(self):
        return "instance of {}(Config)".format(self.__class__)

    @abc.abstractmethod
    def validate(self):
        """
        Validate the config.

        Raises:
            InvalidConfigError: if the config is invalid
        """

        pass


class InvalidConfigError(Exception):
    pass


class Monitor(ABC):
    """
    Monitor component of the model.

    Monitor is a base class for components that store or freeze state for later usage but don't modify it or return any new state objects.

    TODO: Named after Sympl Monitor component: https://sympl.readthedocs.io/en/latest/monitors.html
    """

    def __str__(self):
        return "instance of {}(Monitor)".format(self.__class__)

    @abc.abstractmethod
    def store(self, state, **kwargs):
        """Store state and perform class specific actions on it.


        Args:
            state: dict  model state dictionary
        """
        pass

@dataclass
class FieldIoConfig(Config):
    output_interval: str
    start_time: str
    filename_pattern: str
    variables: list[str]
    

@dataclass
class IoConfig(Config):
    """
    Structured config for IO.

    TODO: (halungge) add
    - vertical output type model levels or pressure levels
    - interval etc per field
    -
    """
    base_name: str
    
    output_end_time: str
    field_configs: list[FieldIoConfig]

    def validate(self):
        pass


class FieldGroupMonitor(Monitor):
    """
    Chain of IO components.
    """

    @property
    def next_output_time(self):
        return self._next_output_time

    @property
    def time_delta(self):
        return self._time_delta
    def __init__(self, config: FieldIoConfig):
        self._next_output_time = datetime.fromisoformat(config.start_time)
        self._time_delta = to_delta(config.output_interval)
        self._field_names = config.filename_pattern
        self._variables = config.variables


    def _update_fetch_times(self):
        self._next_output_time = self._next_output_time + self._time_delta

    def store(self, state, model_time, **kwargs):
        """Pick fields from the state dictionary to be written to disk.


        Args:
            state: dict  model state dictionary
            time: float  model time
        """
        # TODO (halungge) how to handle non time matches? That is if the model time jumps over the output time
        if self._at_capture_time(model_time):
            # this should do a deep copy of the data
            state_to_store = {field: state[field] for field in self._field_names}
            self._update_fetch_times()
            # TODO (halungge) copy data buffer and trigger the processing chain asynchronously?
            
            # trigger processing chain asynchronously

    def _at_capture_time(self, model_time):
        return self._next_output_time == model_time



class IoMonitor(Monitor):
    """
    Monitor implementation for Field IO.
    """

    def __init__(self, config: IoConfig, **kwargs):
        config.validate()
        self._chains = [FieldGroupMonitor(conf) for conf in config.field_configs]
            
       
        
        
        self.config = config
        self.kwargs = kwargs
        
   

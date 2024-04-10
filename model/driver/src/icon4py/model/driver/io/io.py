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

import netCDF4 as nc
import xarray as xr
from dask.delayed import Delayed

from icon4py.model.common.grid.vertical import VerticalModelParams


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

@dataclass(frozen=True)
class FieldIoConfig(Config):
    """
    Structured config for IO of a field group.
    
    Field group is a number of fields that are output at the same time intervals on the same grid 
    (can be any horizontal dimension) and vertical levels.
    
    TODO (halungge) add support for 
    - different vertical levels type config (pressure, model levels, height levels?)
    - regridding (horizontal)
    - end time ?
    """
    output_interval: str
    start_time: str
    filename_pattern: str
    variables: list[str]
    
    def validate(self):
        pass
    

@dataclass(from_dict=True)
class IoConfig(Config):
    """
    Structured and hierarchical config for IO.
    
    Holds some general configuration and a collection of configuraitions for each field group
    """
    base_name: str
    
    output_end_time: str
    field_configs: list[FieldIoConfig]
    #TODO default time units and calendar?

    def validate(self):
        # TODO (halungge) add validation of this configs own fields
        for field_config in self.field_configs:
            field_config.validate()
        


class FieldGroupMonitor(Monitor):
    """
    Monitor for a group of fields.
    
    This monitor is responsible for storing a group of fields that are output at the same time intervals.
    """

    @property
    def next_output_time(self):
        return self._next_output_time

    @property
    def time_delta(self):
        return self._time_delta
    
    
    def __init__(self, config: FieldIoConfig, vertical:VerticalModelParams):
        self._next_output_time = datetime.fromisoformat(config.start_time)
        self._time_delta = to_delta(config.output_interval)
        self._field_names = config.filename_pattern
        self._variables = config.variables
        self._dataset = self._init_dataset(vertical)
        

    #initalise this Monitors dataset with:
    # - global attributes
    # - unlimited time dimension
    # - vertical dimension
    # - horizontal dimensions
    def _init_dataset(self,vertical_grid:VerticalModelParams):
        """ Initialise the dataset with global attributes and dimensions.
        
        TODO (magdalena): as long as we have no terrain it is probably ok to take vct_a as vertical coordinate once there is
        terrain k-heights become [horizontal, vertical ] field
        TODO (magdalena): dimension ordering is # dimension ordering T (ime),Z (height), lat, lon
        TODO (magdalena): for writing a dataset it is best to use netcdf4-python directly, since it has: parallel writing, and
        """
        
        
        attrs = dict(
            Conventions="CF-1.7", # TODO (halungge) check changelog? latest version is 1.11
            title="ICON4Py output", # TODO (halungge) let user in config 
            institution="ETH Zurich and MeteoSwiss",
            source="ICON4Py",
            history="Created by ICON4Py",
            references="https://icon4py.github.io",
            comment="ICON inspired code in Python and GT4Py",
            external_variables="",# TODO (halungge) add list of fields in ugrid file
            uuidOfHGrid="", # TODO (halungge) add uuid of the grid
        )
        df = DatasetFactory(self.config, vertical_grid, attrs)
        self._dataset = df.initialize_dataset()
            

    def _update_fetch_times(self):
        self._next_output_time = self._next_output_time + self._time_delta

    

    def store(self, state, model_time:datetime, **kwargs):
        """Pick fields from the state dictionary to be written to disk.


        Args:
            state: dict  model state dictionary
            time: float  model time
        """
        # TODO (halungge) how to handle non time matches? That is if the model time jumps over the output time
        if self._at_capture_time(model_time):
            # this should do a deep copy of the data
            self._dataset = self._dataset.merge(state)
            state_to_store = {field: state[field] for field in self._field_names}
            self._update_fetch_times()
            # see https: // github.com / pydata / xarray / issues / 1672  # issuecomment-685222909
            
            
            
            # TODO (halungge) copy data buffer and trigger the processing chain asynchronously?
            
            # trigger processing chain asynchronously

    def _at_capture_time(self, model_time):
        return self._next_output_time == model_time



class IoMonitor(Monitor):
    """
    Composite Monitor for all IO Groups
    """

    def __init__(self, config: IoConfig, **kwargs):
        config.validate()
        self.config = config
        
        self._group_monitors = [FieldGroupMonitor(conf) for conf in config.field_configs]
        
    def store(self, state, model_time:datetime, **kwargs):    
        for monitor in self._group_monitors:
            monitor.store(state, model_time, **kwargs)
        


class XArrayNetCDFWriter:
    from xarray import Dataset
    def __init__(self, filename, mode="a"):
        self.filename = filename
        self.mode = mode
        

    def write(self, dataset:Dataset, immediate=True)->[Delayed|None]:
        delayed = dataset.to_netcdf(self.filename, mode=self.mode, engine="netcdf4", format="NETCDF4", unlimited_dims=["time"], compute=immediate)
        return delayed
    def close(self):
        self.dataset.close()
        self.dataset = None
        return self.dataset
    
    
class DatasetFactory:
    def __init__(self, file_name: str, num_lev:int, global_attrs:dict):
        self._file_name = file_name
        self.num_lev = num_lev
        self.attrs = global_attrs
        self.dataset = None

    def add_dimension(self, name: str, values: xr.DataArray, ):
        self.dataset.createDimension('name', values.shape[0])
        self.dataset.createVariable(values.attrs["short"] values.dtype, (name,))
    def initialize_dataset(self):
        # TODO (magdalena) (what mode do we need here?)
        self.dataset = nc.Dataset(self._file_name, "w", format="NETCDF4")
        self.dataset.setncatts(self.attrs)
        ## create dimensions all except time are fixed
        
        self.dataset.createDimension('time', None)
        self.dataset.createDimension('height', self.vertical.num_lev) 
        self.dataset.createDimension('mass_level', self.vertical.num_lev)
        self.dataset.createDimension('interface_level', self.vertical.num_lev+1)
        # create coordinate variables
        times = self.dataset.createVariable('times', 'f8', ('time',))
        times.units = 'seconds since 1970-01-01 00:00:00'
        
        
        # create dimensions
        
        # xarray : 
        #time = DataArray([],name="time", attrs=dict(standard_name="time", long_name="time", units="seconds since 1970-01-01 00:00:00", calendar=DEFAULT_CALENDAR))
        #height = to_data_array(vertical_grid.vct_a, attrs=dict(standard_name="height", long_name="height", units="m"))
        #mass_levels = np.range(vertical_grid.num_levels, dtype=np.int32)
        #half_levels = np.range(vertical_grid.num_levels+1, dtype=np.int32)
        #interface_levels = DataArray(half_levels, attrs=dict(standard_name="model level number", long_name="model interface levels", units="1"))
        #full_levels = DataArray(mass_levels, attrs=dict(standard_name="model_level_number", long_name="mass levels", units="1"))
        #coord_vars = dict(time=time, mass_level=full_levels, interface_level=interface_levels, height=height) 
        #self._dataset = Dataset(data_vars=None, coords=coord_vars, attrs=attrs)
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
import logging
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import xarray as xr
from dask.delayed import Delayed

from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.driver.io.xgrid import IconUGridWriter


log = logging.getLogger(__name__)

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
    

@dataclass
class IoConfig(Config):
    """
    Structured and hierarchical config for IO.
    
    Holds some general configuration and a collection of configuraitions for each field group
    
    """
    base_name: str # base name for all outputfiles
    
    output_end_time: str #: end time for output
    time_units = "seconds since 1970-01-01 00:00:00"
    calendar = "proleptic_gregorian"
    output_path:str
    field_configs: list[FieldIoConfig]
    

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
    
    
    def __init__(self, config: FieldIoConfig, vertical:VerticalGridSize, horizontal:HorizontalGridSize):
        self._next_output_time = datetime.fromisoformat(config.start_time)
        self._time_delta = to_delta(config.output_interval)
        self._field_names = config.variables
        self._file_name = config.filename_pattern
        self._init_dataset(vertical, horizontal)
        

    #initalise this Monitors dataset with:
    # - global attributes
    # - unlimited time dimension
    # - vertical dimension
    # - horizontal dimensions
    def _init_dataset(self,vertical_grid:VerticalGridSize, horizontal_size:HorizontalGridSize):
        """ Initialise the dataset with global attributes and dimensions.
        
        TODO (magdalena): as long as we have no terrain it is probably ok to take vct_a as vertical coordinate once there is
        terrain k-heights become [horizontal, vertical ] field
        TODO (magdalena): dimension ordering is # dimension ordering T (ime),Z (height), lat, lon
        TODO (magdalena): for writing a dataset it is best to use netcdf4-python directly, since it has: parallel writing, and
        """
        
        #TODO (magdalena) common parts should be constructed by IoMonitor
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
        # can this be an xarray dataset?
        df = DatasetStore(self.config, vertical_grid, horizontal_size, attrs)
        self._dataset = df
            

    def _update_fetch_times(self):
        self._next_output_time = self._next_output_time + self._time_delta

    

    def store(self, state:dict, model_time:datetime, **kwargs):
        """Pick fields from the state dictionary to be written to disk.

        # TODO: this should be a Listener pattern, that copy data buffer and trigger the rest chain asynchronously
        Args:
            state: dict  model state dictionary
            time: float  model time
        """
        # TODO (halungge) how to handle non time matches? That is if the model time jumps over the output time
        if self._at_capture_time(model_time):
            # this should do a deep copy of the data
            state_to_store = {field: state[field] for field in self._field_names}
            self._update_fetch_times()
            self._append_data(state_to_store, model_time)
            
            # see https: // github.com / pydata / xarray / issues / 1672  # issuecomment-685222909
            
            
            
           
    def _append_data(self, state_to_store:dict, model_time:datetime):
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self, model_time):
        return self._next_output_time == model_time




class IoMonitor(Monitor):
    """
    Composite Monitor for all IO Groups
    """

    def __init__(self, config: IoConfig, grid_file:str):
        config.validate()
        self.config = config
        self._grid_file = grid_file
        self._group_monitors = [FieldGroupMonitor(conf) for conf in config.field_configs]
        self._initialize_output()
        
        
    def _initialize_output(self):
        self._create_output_dir()
        self._write_ugrid()
        
        
    
    def _create_output_dir(self):
        path= Path(self.config.output_path)
        try: 
            path.mkdir(parents=True, exist_ok=False)
            self._output_path = path
        except OSError as error:
            log.error(f"Output directory at {path} exists: {error}.")
            
        
        
    def _write_ugrid(self):
        writer = IconUGridWriter(self._grid_file, self._output_path)
        writer(validate=True)
        
    @property    
    def path(self):
        return self._output_path
        
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
    
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
class DatasetStore:
    def __init__(self, file_name: str, vertical:VerticalGridSize, horizontal:HorizontalGridSize ,global_attrs:dict):
        self._file_name = file_name
        self._counter = 1
        self.num_levels = vertical.num_lev
        self.horizontal_size = horizontal
        self.attrs = global_attrs
        self.dataset = None
 
    def __getitem__(self, item):
        return self.dataset.getncattr(item)

    def add_dimension(self, name: str, values: xr.DataArray ):
        self.dataset.createDimension(name, values.shape[0])
        dim = self.dataset.createVariable(name,  values.dtype, (name,))
        dim.units = values.units
        if hasattr(values.attrs,'calendar'):
            dim.calendar = values.calendar
            
        dim.long_name = values.long_name
        dim.standard_name = values.standard_name
        
    def initialize_dataset(self):
        # TODO (magdalena) (what mode do we need here?)
        filename = generate_name(self._file_name, self._counter)
        self.dataset = nc.Dataset(filename, "w", format="NETCDF4")
        self.dataset.setncatts(self.attrs)
        ## create dimensions all except time are fixed
        
        self.dataset.createDimension('time', None)
        self.dataset.createDimension('full_level', self.num_levels) 
        self.dataset.createDimension('interface_level', self.num_levels+1)
        self.dataset.createDimension('cell', self.horizontal_size.num_cells)
        self.dataset.createDimension('vertex', self.horizontal_size.num_vertices)
        self.dataset.createDimension('edge', self.horizontal_size.num_edges)
        # create time variables
        times = self.dataset.createVariable('times', 'f8', ('time',))
        times.units = 'seconds since 1970-01-01 00:00:00'
        times.calendar = 'proleptic_gregorian'
        times.standard_name = 'time'
        times.long_name = 'time'
        # create vertical coordinates
        

    def append(self, state:dict, model_time:datetime):
       
        #TODO (magdalena) add time to the dataset
        #TODO (magdalena) add data to the dataset
        pass

    @property
    def dims(self) -> dict:
        return self.dataset.dimensions
    @property
    def variables(self) -> dict:
        return self.dataset.variables
        
        
def generate_name(fname:str, counter:int)->str:
    stem = fname.split(".")[0]
    return f"{stem}_{counter}.nc"
    
    

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
from typing import Sequence

import netCDF4 as nc
import numpy as np
import xarray
import xarray as xr
from cftime import date2num
from dask.delayed import Delayed

from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.driver.io.cf_utils import (
    COARDS_T_POS,
    DEFAULT_CALENDAR,
    DEFAULT_TIME_UNIT,
    INTERFACE_LEVEL_NAME,
    LEVEL_NAME,
)
from icon4py.model.driver.io.xgrid import IconUGridWriter, load_data_file


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
    - end time ?
    """
    output_interval: str
    start_time: str
    filename_pattern: str
    variables: list[str]
    title="ICON4Py Simulation"
    comment="ICON inspired code in Python and GT4Py"
    
    def validate(self):
        assert self.output_interval, "Output interval is not set."
        assert self.start_time, "Start time is not set."
        assert self.filename_pattern, "Filename pattern is not set."
        assert self.variables, "No variables provided for output."





@dataclass(frozen=True)
class IoConfig(Config):
    """
    Structured and hierarchical config for IO.

    Holds some general configuration and a collection of configuraitions for each field group

    """
    output_path: str = './output/'
    base_name: str = 'icon4py_output_'
    field_configs: Sequence[FieldIoConfig] = ()

    time_units = DEFAULT_TIME_UNIT
    calendar = DEFAULT_CALENDAR

    def validate(self):
        if not self.field_configs:
            log.warning("No field configurations provided for output")
        else:
            for field_config in self.field_configs:
                field_config.validate()

class IoMonitor(Monitor):
    """
    Composite Monitor for all IO Groups
    """

    def __init__(self, config: IoConfig, vertical_size: VerticalGridSize, horizontal_size:HorizontalGridSize, 
                 grid_file_name: str, grid_id:str):
        config.validate()
        self.config = config
        self._grid_file = grid_file_name
        self._group_monitors = [FieldGroupMonitor(conf,vertical=vertical_size, horizontal=horizontal_size, grid_id=grid_id) for conf in
                                config.field_configs]
        self._initialize_output()

    def _read_grid_attrs(self) -> dict:
        with load_data_file(self._grid_file) as ds:
            return ds.attrs

    def _initialize_output(self):
        self._create_output_dir()
        self._write_ugrid()

    def _create_output_dir(self):
        path = Path(self.config.output_path)
        try:
            path.mkdir(parents=True, exist_ok=False, mode=0o777)
            self._output_path = path
        except OSError as error:
            log.error(f"Output directory at {path} exists: {error}.")

    def _write_ugrid(self):
        writer = IconUGridWriter(self._grid_file, self._output_path)
        writer(validate=True)

    @property
    def path(self):
        return self._output_path

    def store(self, state, model_time: datetime, **kwargs):
        for monitor in self._group_monitors:
            monitor.store(state, model_time, **kwargs)





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
    
    
    def __init__(self, config: FieldIoConfig, vertical:VerticalGridSize, horizontal:HorizontalGridSize, grid_id:str):

        self._global_attrs = dict(
            Conventions="CF-1.7",  # TODO (halungge) check changelog? latest version is 1.11
            title=config.title,
            comment=config.comment,
            institution="ETH Zurich and MeteoSwiss",
            source="ICON4Py",
            history="Created by ICON4Py",
            references="https://icon4py.github.io",
            external_variables="",  # TODO (halungge) add list of fields in ugrid file
            uuidOfHGrid=grid_id,  
        )
        self._next_output_time = datetime.fromisoformat(config.start_time)
        self._time_delta = to_delta(config.output_interval)
        self._field_names = config.variables
        self._file_name = config.filename_pattern
        self._init_dataset(vertical, horizontal)
        
        

    #initialise this Monitors dataset with:
    # - global attributes
    # - unlimited time dimension -> time
    # - vertical dimension(s)
    # - horizontal dimensions
    def _init_dataset(self, vertical_grid:VerticalGridSize, horizontal_size:HorizontalGridSize):
        """ Initialise the dataset with global attributes and dimensions.
        
        TODO (magdalena): as long as we have no terrain it is probably ok to take vct_a as vertical coordinate once there is
        terrain k-heights become [horizontal, vertical ] field
        TODO (magdalena): dimension ordering is # dimension ordering T (ime),Z (height), lat, lon
        TODO (magdalena): for writing a dataset it is best to use netcdf4-python directly, since it has: parallel writing, and
        """
        
        #TODO (magdalena) common parts should be constructed by IoMonitor
        
        df = NetcdfWriter(self._file_name, vertical_grid, horizontal_size, self._global_attrs)
        df.initialize_dataset()
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
            # see https://github.com/pydata/xarray/issues/1672
            
            
            
           
    def _append_data(self, state_to_store:dict, model_time:datetime):
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self, model_time):
        return self._next_output_time == model_time








    
class NetcdfWriter:
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
        # TODO (magdalena) (what mode do we need `a` or `w`? 
        #  TODO properly closing file
        filename = generate_name(self._file_name, self._counter)
        self.dataset = nc.Dataset(filename, "w", format="NETCDF4")
        self.dataset.setncatts(self.attrs)
        ## create dimensions all except time are fixed
        
        self.dataset.createDimension('time', None)
        self.dataset.createDimension('level', self.num_levels) 
        self.dataset.createDimension('interface_level', self.num_levels+1)
        self.dataset.createDimension('cell', self.horizontal_size.num_cells)
        self.dataset.createDimension('vertex', self.horizontal_size.num_vertices)
        self.dataset.createDimension('edge', self.horizontal_size.num_edges)
        # create time variables
        times = self.dataset.createVariable('times', 'f8', ('time',))
        times.units = DEFAULT_TIME_UNIT
        times.calendar = DEFAULT_CALENDAR
        times.standard_name = 'time'
        times.long_name = 'time'
        # create vertical coordinates:
        levels = self.dataset.createVariable('levels', np.int32, ('level',))
        levels.units = '1'
        levels.long_name = 'model full levels'
        levels.standard_name = LEVEL_NAME
        levels[:] = np.arange(self.num_levels, dtype=np.int32)
        
        interface_levels = self.dataset.createVariable('interface_levels', np.int32, ('interface_level',))
        interface_levels.units = '1'
        interface_levels.long_name = 'model interface levels'
        interface_levels.standard_name = INTERFACE_LEVEL_NAME
        interface_levels[:] = np.arange(self.num_levels+1, dtype=np.int32)
        
        # TODO (magdalena) add vct_a as vertical coordinate?

    def append(self, state_to_append:dict[str, xarray.DataArray], model_time:datetime):
       
        #TODO (magdalena) add data to the dataset
        #3. if yes find location of the time variable
        #4. append the data
        time = self.dataset["times"]
        time_pos = len(time)
        time[time_pos] = date2num(model_time, units=time.units, calendar=time.calendar)
        for k, new_slice in state_to_append.items():
            standard_name = new_slice.standard_name
            
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                # TODO: 
                #spatial_dims, reorder = self._to_canonical_dim_order(new_slice.dims)
                dimensions = ('time',) + new_slice.dims
                new_var = self.dataset.createVariable(k, new_slice.dtype, dimensions)
                new_var[0, :] = new_slice.data
                new_var.units = new_slice.units
                new_var.standard_name = new_slice.standard_name
                new_var.long_name = new_slice.long_name
                new_var.coordinates = new_slice.coordinates
                
            else:
                var_name = ds_var.get(k).name
                dims = ds_var.get(k).dimensions
                shape = ds_var.get(k).shape
                assert len(new_slice.dims) == len(dims) -1 , f"Data variable dimensions do not match for {standard_name}."

                # TODO (magdalena) this needs to be changed for distributed case where we write to global_index slice for the horizontal dim.
                # we can acutally assume fixed index ordering here, input arrays should re shaped to canonical order
                
                right = (slice(None),) * (len(dims) - 1)
                expand_slice = (slice(shape[COARDS_T_POS] - 1, shape[COARDS_T_POS]),)
                slices = expand_slice + right
                self.dataset.variables[var_name][slices] = new_slice.data
                #self.append_data(k, v, time_pos)
        
        
    def close(self):
        self.dataset.close()
        
    @property
    def dims(self) -> dict:
        return self.dataset.dimensions
    @property
    def variables(self) -> dict:
        return self.dataset.variables

    def _to_canonical_dim_order(self, dims:tuple[str,...]):
        """Check for dimension being in canoncial order ('T', 'Z', 'Y', 'X') and return them in this order.
        
        
        """
        pass


class XArrayNetCDFWriter:
    from xarray import Dataset
    def __init__(self, filename, mode="a"):
        self.filename = filename
        self.mode = mode

    def write(self, dataset: Dataset, immediate=True) -> [Delayed | None]:
        delayed = dataset.to_netcdf(self.filename, mode=self.mode, engine="netcdf4",
                                    format="NETCDF4", unlimited_dims=["time"], compute=immediate)
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


def generate_name(fname:str, counter:int)->str:
    stem = fname.split(".")[0]
    return f"{stem}_{counter}.nc"
    
    
def filter_by_standard_name(model_state:dict, value:str):
    return {k:v for k,v in model_state.items() if value == v.standard_name}
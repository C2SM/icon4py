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
import logging
import numpy as np
from datetime import datetime, timedelta

from gt4py.next.common import Field
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionConfig, DiffusionParams
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.icon_grid import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.dimension import (
    CellDim,
    EdgeDim,
    KDim
)

log = logging.getLogger(__name__)


class ModelTime:

    def __init__(self,
        start_date: datetime = datetime(0, 0, 0, 0, 0),
        nsteps: int = 0,  # number of time steps to be integrated
        dtime: float = 0.0, # time step, in seconds
        # add physics time steps here
        ndyn_substeps: int = 5 # number of substeps in dynamics
    ):
        # basic read from input arguments
        self.__start_date: datetime = start_date
        self.__nsteps: int = nsteps
        self.dtime: float = dtime
        self.ndyn_substeps: int = ndyn_substeps

        self.now_date: datetime = start_date
        self.now_step: int = 0

    @property
    def start_date(self):
        return self.__start_date

    @property
    def nsteps(self):
        return self.__nsteps

    def next_step(self):
        self.now_step += 1
        self.now_date = self.now_date + timedelta(seconds=self.dtime)


class ModelTimer:
    # timer class for recording and measuring the process time lapsed
    pass

class PrognosticClass:

    def __init__(self,
        cell_size: int = 1,
        edge_size: int = 3,
        k_size: int = 1
    ):
        # vertical_wind field,  w(nproma, nlevp1, nblks_c) [m/s]
        self.w: Field[[CellDim, KDim], float] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float))
        # vn(nproma, nlev, nblks_e)  [m/s]
        self.vn: Field[[EdgeDim, KDim], float] = np_as_located_field(EdgeDim,KDim)(np.zeros((edge_size,k_size),dtype=float))
        # exner(nrpoma, nlev, nblks_c)
        self.exner: Field[[CellDim, KDim], float] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float))
        # (nproma, nlev, nlbks_c) [K]
        self.theta_v: Field[[CellDim, KDim], float] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float))



class DiagnosticClass:

    def __init__(self,
        cell_size: int = 1,
        edge_size: int = 3,
        k_size: int = 1
    ):
        self.airmass: Field[[CellDim, KDim], float] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size, k_size), dtype=float))

class NHstate:

    def __init__(self,
        cell_size: int = 1,
        edge_size: int = 3,
        k_size: int = 1
    ):
        self.cell_size: int = cell_size
        self.edge_size: int = edge_size
        self.k_size: int = k_size
        self.prognostic_var_now: PrognosticClass = PrognosticClass(cell_size,edge_size,k_size)
        self.prognostic_var_new: PrognosticClass = PrognosticClass(cell_size,edge_size,k_size)
        self.prognostic_var_old: PrognosticClass = PrognosticClass(cell_size,edge_size,k_size)
        self.diagnostic_var_now: DiagnosticClass = DiagnosticClass(cell_size,edge_size,k_size)
        self.diagnostic_var_new: DiagnosticClass = DiagnosticClass(cell_size,edge_size,k_size)
        self.diagnostic_var_old: DiagnosticClass = DiagnosticClass(cell_size,edge_size,k_size)


    def compute_airmass_now(self):
        # compute airmass
        pass

class Model_driver:

    def __init__(self,
        time: ModelTime,
        model_var: NHstate,
        grid: IconGrid
    ):
        self.time: ModelTime = time
        self.model_var: NHstate = model_var
        self.grid: IconGrid = grid
        self._initialized: bool = False
        self.diffusion: Diffusion = None
        self.diffusion_config: DiffusionConfig = None
        self.diffusion_params: DiffusionParams = None
        self.interpolation_state = None
        self.vertical_param = None
        self.edge_param = None
        self.cell_param = None
        self.metric_state = None


    def model_initialization(self):
        # some initializations to be done here
        diffusion = Diffusion()

        diffusion.init(
            grid=self.grid,
            config=self.diffusion_config,
            params=self.diffusion_params,
            vertical_params=self.vertical_param,
            metric_state=self.metric_state,
            interpolation_state=self.interpolation_state,
            edge_params=self.edge_param,
            cell_params=self.cell_param
        )

        self._initialized = True

    def time_loop(self):

        # execution of time loop
        for time_step in range(self.time.nsteps):

            # set boundary conditions (not required in JW test)

            # perform dynamical calculations
            self._integrate_nh()

            # diagnostic calculations

            # some IO

            self.time.next_step()



    def _integrate_nh(self):

        self._dycore_substepping()

        # call diffusion

        # tracer advection


    def _dycore_substepping(self):

        # diagnostic calculations
        # airmass

        for i in range(self.time.ndyn_substeps):
            # call diffusion
            # call solv_nh driver
            pass

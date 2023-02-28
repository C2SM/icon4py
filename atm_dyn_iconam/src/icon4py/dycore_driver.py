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

from dataclasses import dataclass

from icon4py.diffusion.diffusion import Diffusion


@dataclass
class AtmoNonHydroConfig:
    n_substeps: int


class AtmoNonHydro:
    def __init__(self, config):
        self.config = config

    def _dynamics_timestep(self):
        pass

    def do_dynamics_substepping(self):
        for i in range(self.config.n_substeps):
            self._dynamics_timestep()


diffusion: Diffusion
non_hydro: AtmoNonHydro


def timestep(dtime: float):
    diffusion.initial_step()
    non_hydro.do_dynamics_substepping()
    diffusion.time_step()


def timeloop(dtime: float):
    pass

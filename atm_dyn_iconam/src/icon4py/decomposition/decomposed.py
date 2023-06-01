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

import mpi4py.MPI


class ProcessProperties:
    def __init__(self,  name="", size = 0, rank=0):
        self._communicator_name: str = name
        self._rank: int = rank
        self._comm_size = size

    @property
    def rank(self):
        return self._rank

    @property
    def comm_name(self):
        return self._communicator_name
    @property
    def comm_size(self):
        return self._comm_size

    @classmethod
    def from_mpi_comm(cls, comm: mpi4py.MPI.Comm):
        return ProcessProperties(comm.Get_name(), comm.Get_rank())

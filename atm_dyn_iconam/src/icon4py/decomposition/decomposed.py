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
    def __init__(self, comm: mpi4py.MPI.Comm):
        self._communicator_name: str = comm.Get_name()
        self._rank: int = comm.Get_rank()
        self._comm_size = comm.Get_size()
        self._comm = comm

    @property
    def rank(self):
        return self._rank

    @property
    def comm_name(self):
        return self._communicator_name

    @property
    def comm_size(self):
        return self._comm_size

    @property
    def comm(self):
        return self._comm

    @classmethod
    def from_mpi_comm(cls, comm: mpi4py.MPI.Comm):
        return ProcessProperties(comm)

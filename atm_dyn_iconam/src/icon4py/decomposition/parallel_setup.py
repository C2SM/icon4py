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
from typing import Union

import mpi4py
from mpi4py.MPI import Comm



mpi4py.rc.initialize = False

CommId = Union[int, Comm, None]
log = logging.getLogger(__name__)


def get_processor_properties(comm_id: CommId = None):
    init_mpi()

    def _get_current_comm_or_comm_world(comm_id: CommId) -> Comm:
        if isinstance(comm_id, int):
            comm = Comm.f2py(comm_id)
        elif isinstance(comm_id, Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    current_comm = _get_current_comm_or_comm_world(comm_id)
    return ProcessProperties.from_mpi_comm(current_comm)


def init_mpi():
    from mpi4py import MPI

    if not MPI.Is_initialized():
        log.info("initializing MPI")
        MPI.Init()


def finalize_mpi():
    from mpi4py import MPI

    if not MPI.Is_finalized():
        log.info("finalizing MPI")
        MPI.Finalize()


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

class ParallelLogger(logging.Filter):
    def __init__(self, processProperties: ProcessProperties = None):
        super().__init__()
        self._rank_info = ""
        if processProperties and processProperties.comm_size > 1:
            self._rank_info = f"rank={processProperties.rank}/{processProperties.comm_size} [{processProperties.comm_name}] >>>"

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank_info
        return True

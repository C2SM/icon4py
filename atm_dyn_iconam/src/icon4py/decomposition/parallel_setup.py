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
import functools
import logging
from dataclasses import dataclass
from typing import Optional, Union

import mpi4py
from mpi4py.MPI import Comm


mpi4py.rc.initialize = False

CommId = Union[int, Comm, None]
log = logging.getLogger(__name__)


def get_processor_properties(with_mpi=False, comm_id: CommId = None):
    def _get_current_comm_or_comm_world(comm_id: CommId) -> Comm:
        if isinstance(comm_id, int):
            comm = Comm.f2py(comm_id)
        elif isinstance(comm_id, Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    if with_mpi:
        init_mpi()
        current_comm = _get_current_comm_or_comm_world(comm_id)
        return ProcessProperties.from_mpi_comm(current_comm)
    else:
        return ProcessProperties.from_single_node()


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

@dataclass(frozen=True)
class ProcessProperties:
    comm: Optional[mpi4py.MPI.Comm] = None


    @functools.cached_property
    def rank(self):
        return self.comm.Get_rank() if self.comm else 0


    @functools.cached_property
    def comm_name(self):
        return self.comm.Get_name() if self.comm else ""

    @functools.cached_property
    def comm_size(self):
        return self.comm.Get_size() if self.comm else 1


    @classmethod
    def from_mpi_comm(cls, comm: mpi4py.MPI.Comm):
        return ProcessProperties(comm)

    @classmethod
    def from_single_node(cls):
        return ProcessProperties()


class ParallelLogger(logging.Filter):
    def __init__(self, processProperties: ProcessProperties = None):
        super().__init__()
        self._rank_info = ""
        if processProperties and processProperties.comm_size > 1:
            self._rank_info = f"rank={processProperties.rank}/{processProperties.comm_size} [{processProperties.comm_name}] "

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank_info
        return True

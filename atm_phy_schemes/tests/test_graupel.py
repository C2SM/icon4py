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

import os
from sys import exit, stderr

import numpy as np

import serialbox as ser


# Configuration
SER_DATA = os.path.join(os.path.dirname(__file__), "ser_data")
TEST = True
WRITE_RESULTS = False
NoMPIRanks = 6


def test_graupel():

    if not os.path.exists(SER_DATA):
        os.system("./get_data.sh")

    for rank in range(1, NoMPIRanks + 1):
        print("=======================")
        print(f"Runing rank {str(rank)}")

        try:
            serializer = ser.Serializer(
                ser.OpenModeKind.Read, SER_DATA, "ref_rank_" + str(rank)
            )
        except ser.SerialboxError as e:
            print("serializer: error: {}".format(e), file=stderr)
            exit(1)

        savepoints = serializer.savepoint_list()

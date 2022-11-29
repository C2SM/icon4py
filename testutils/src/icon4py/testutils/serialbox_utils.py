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
from typing import List


try:
    import serialbox as ser
except ImportError:
    external_src = os.path.join(os.path.dirname(__file__), "../../_external_src/")
    os.chdir(external_src)
    os.system(
        "git clone --recursive https://github.com/GridTools/serialbox; CC=`which gcc` CXX=`which g++` pip install serialbox/src/serialbox-python"
    )
    import serialbox as ser


def read_from_call_diffusion_init_ser_data(
    path, fname_prefix, metadata: List[str], fields: List[str]
):
    rank = 0
    fname = f"{fname_prefix}_rank{str(rank)}"
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, fname)
    save_points = serializer.savepoint_list()
    print(save_points)
    field_names = serializer.fieldnames()
    print(field_names)
    savepoint = (
        serializer.savepoint["call-diffusion-init"]
        .linit[False]
        .date["2021-06-20T12:00:10.000"]
        .as_savepoint()
    )
    print(savepoint.metainfo)
    meta_present = {}
    meta_absent = []
    for md in metadata:
        if md in savepoint.metainfo.to_dict():
            meta_present[md] = savepoint.metainfo[md]
        else:
            meta_absent.append(md)

    fields_present = {}
    fields_absent = []
    for field_name in fields:
        if field_name in field_names:
            fields_present[field_name] = serializer.read(field_name, savepoint)
        else:
            fields_absent.append(field_name)
    [print(f"field  {f} not present in savepoint") for f in fields_absent]
    [print(f"metadata  {f} not present in savepoint") for f in meta_absent]

    return fields_present, meta_present

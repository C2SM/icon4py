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


def read_from_ser_data(path, metadata: List[str], fields: List[str]):
    rank = 0
    serializer = ser.Serializer(
        ser.OpenModeKind.Read, path, f"reference_icon_rank{str(rank)}"
    )
    save_points = serializer.savepoint_list()
    print(save_points)
    field_names = serializer.fieldnames()
    print(field_names)
    savepoint = serializer.savepoint["diffusion-in"].id[0].as_savepoint()
    print(type(savepoint))
    print(savepoint)
    meta_present = {}
    meta_absent = []
    for md in metadata:
        if md in savepoint.metainfo:
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

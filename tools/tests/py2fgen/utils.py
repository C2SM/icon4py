# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import sysconfig


def get_prefix_lib_path():
    lib_folder = sysconfig.get_config_vars().get("LIBDIR").split("/")[-1]
    rpath = f"{sys.base_prefix}/{lib_folder}"
    return rpath

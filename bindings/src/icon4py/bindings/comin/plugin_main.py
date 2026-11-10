# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script that ComIn's Python adapter runs as the icon4py plugin's primary constructor.

ICON loads it with one namelist group (absolute paths):

    &comin_plugin_nml
     name           = "icon4py"
     plugin_library = "<build>/externals/comin/build/_icon/plugins/python_adapter/libpython_adapter.so"
     options        = "<icon4py>/bindings/src/icon4py/bindings/comin/plugin_main.py"
    /
"""

from icon4py.bindings.comin import plugin


plugin.register()

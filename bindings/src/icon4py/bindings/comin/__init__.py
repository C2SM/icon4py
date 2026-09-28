# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
ComIn plugin that runs icon4py granules for ICON (EXCLAIM).

ICON's ComIn backend of the icon4py interface exposes every argument of the py2fgen-exported
functions as ComIn data; the plugin ('plugin.py', loaded through 'plugin_main.py') calls the
undecorated functions with zero-copy views ('_marshal.py', '_views.py'); '_diagnostics.py'
serves its timing mode. The ComIn module 'comin' exists only inside ICON; nothing in this
package imports it at module level.
"""

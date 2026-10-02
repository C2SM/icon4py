# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
ComIn plugin that runs icon4py granules for ICON (EXCLAIM).

With 'icon4py_interface=1' ICON delegates its horizontal diffusion to this plugin ('plugin.py',
loaded through 'plugin_main.py'), which builds the arguments of py2fgen's exported functions
from ComIn's native data (ICON's namelist output, '_config.py'; ComIn's descriptive data,
'_descrdata.py'; ICON's variables) and calls the undecorated functions with zero-copy views
('_arguments.py', '_views.py'). In VERIFY it compares its results with ICON's in ICON's table
format ('_verify.py'); '_diagnostics.py' serves its timing mode; '_probe.py' compares the
arguments the plugin would hand the granules with py2fgen's in a run that computes through
py2fgen (the py2fgen probe), with the comparisons of '_dual.py'. The ComIn module 'comin'
exists only inside ICON; nothing in this package imports it at module level.
"""

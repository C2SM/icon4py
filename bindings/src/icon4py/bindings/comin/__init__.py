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
loaded through 'plugin_main.py'), which builds icon4py's grid and diffusion granule through
icon4py's API ('_granule.py') from ComIn's native data (ICON's namelist output, '_config.py';
ComIn's descriptive data, '_descrdata.py'; ICON's variables as zero-copy views,
'_arguments.py', '_views.py'). In VERIFY it compares its results with ICON's in ICON's table
format ('_verify.py'); '_diagnostics.py' serves its timing mode. The py2fgen probe
('_probe.py', with the comparisons of '_compare.py') is a diagnostic, imported only in probe
mode: in a run that computes through py2fgen it compares the inputs the plugin would hand the
granule, and the grid and granule it builds, with py2fgen's. The ComIn module 'comin' exists
only inside ICON; nothing in this package imports it at module level.
"""

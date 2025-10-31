# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
VizTracer plugin for PY2FGEN generated bindings.

To enable this plugin
- install viztracer ('pip install viztracer')
- set the environment variable PY2FGEN_EXTRA_MODULES=icon4py.tools.py2fgen.wrappers.viztracer_plugin
"""

import warnings

from icon4py.tools.py2fgen import runtime_config


try:
    import viztracer  # type: ignore[import-not-found]

    counter = 0
    tracer = viztracer.VizTracer()

    def viztracer_plugin_enable(_: str) -> None:
        global counter  # noqa: PLW0603 # modifying global variable
        if counter == 0:
            tracer.start()
        counter += 1

    def viztracer_plugin_disable(_: str) -> None:
        if counter == 12:
            tracer.stop()
            tracer.save("viztracer_output.json")

    runtime_config.HOOK_FUNCTION_ENTER = viztracer_plugin_enable
    runtime_config.HOOK_FUNCTION_EXIT = viztracer_plugin_disable

except ImportError:
    warnings.warn("viztracer is not installed; viztracer_plugin is not enabled.", stacklevel=1)

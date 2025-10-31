# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from icon4py.tools.py2fgen import runtime_config


try:
    import viztracer  # type: ignore[import-not-found]

    counter = 0

    def viztracer_plugin_enable(_: str) -> None:
        viztracer.start()
        global counter  # noqa: PLW0603 # modifying global variable
        counter += 1

    def viztracer_plugin_disable(_: str) -> None:
        if counter == 12:
            viztracer.stop()
            viztracer.save("viztracer_output.json")

    runtime_config.HOOK_FUNCTION_ENTER = viztracer_plugin_enable
    runtime_config.HOOK_FUNCTION_EXIT = viztracer_plugin_disable

except ImportError:
    warnings.warn("viztracer is not installed; viztracer_plugin will not work.", stacklevel=1)

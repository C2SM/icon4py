# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Mypy plugin for icon4py.

Usage::

    ```toml
    # pyproject.toml
    [tool.mypy]
    plugins = ['gt4py.next.type_system.mypy_plugin', 'icon4py.model.common.type_system.mypy_plugin']
    ```

The goal of this plugin is to reduce false positives from mypy that arise from
icon4py's runtime-mutable precision aliases (``vpfloat`` in
:mod:`icon4py.model.common.type_alias`).  ``vpfloat`` is reassigned at runtime
by :func:`set_precision` to switch between single and double precision, so it
is not a fixed :data:`~typing.TypeAlias` — mypy therefore reports
``valid-type`` when it is used in type annotations.  This plugin mirrors the
existing GT4Py ``blur_float_precision`` hook: whenever ``vpfloat`` appears in a
type expression, it is resolved to ``builtins.float`` (the GT4Py dsl will catch
actual precision mismatches at runtime).
"""

from __future__ import annotations

import typing


try:
    from mypy import plugin as mplugin, types

    def blur_vpfloat_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """Resolve ``vpfloat`` to ``builtins.float`` in type positions."""
        return ctx.api.named_type("builtins.float", [])

    class Icon4PyPlugin(mplugin.Plugin):
        def get_type_analyze_hook(
            self, fullname: str
        ) -> typing.Callable[[mplugin.AnalyzeTypeContext], types.Type] | None:
            if fullname == "icon4py.model.common.type_alias.vpfloat":
                return blur_vpfloat_precision
            return None

    def plugin(version: str) -> type[mplugin.Plugin]:
        return Icon4PyPlugin

except ImportError:
    pass

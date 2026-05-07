# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-string render API for py2fgen.

Produces the four bindings sources (.py, .f90, .h, .c) as in-memory strings
without touching disk. Callers (the CLI, ``all_bindings``, custom build
scripts) decide what to write, where.

CFFI's ``emit_c_code`` accepts any writable stream; we hand it a
``StringIO`` to keep the renderer I/O-free.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, cast

import cffi

from icon4py.tools.py2fgen import _codegen


@dataclass(frozen=True)
class RenderedSources:
    """The four bindings artifacts produced by :func:`render`, all as strings."""

    py: str
    f90: str
    h: str
    c: str


def render(
    plugin: _codegen.BindingsLibrary,
    *,
    h_basename: str,
    rpath: str = "",
) -> RenderedSources:
    """Render every bindings source for ``plugin`` as an in-memory string.

    Args:
        plugin: The parsed bindings description (from
            :func:`icon4py.tools.py2fgen._generator.get_cffi_description`).
        h_basename: Basename embedded in the ``#include "<h_basename>"``
            line at the top of the generated C source. The caller picks
            this so the eventual on-disk ``.h`` filename matches.
        rpath: Optional runtime library search path to embed in
            ``extra_link_args``. Only relevant when the rendered ``.c`` is
            later compiled.
    """
    py_wrapper = _codegen.generate_python_wrapper(plugin)
    f90_interface = _codegen.generate_f90_interface(plugin)
    c_header = _codegen.generate_c_header(plugin)

    builder = cffi.FFI()
    extra_link_args = [f"-Wl,-rpath={rpath}"] if rpath else []
    builder.embedding_api(c_header)
    builder.set_source(
        plugin.library_name, f'#include "{h_basename}"', extra_link_args=extra_link_args
    )
    builder.embedding_init_code(py_wrapper)

    buf = io.StringIO()
    # CFFI's emit_c_code accepts any writable stream at runtime (long-stable
    # behavior; cffi>=1.5 is pinned in bindings/pyproject.toml). Its type stub
    # only declares ``filename: str``, so we narrow the suppression to a cast.
    builder.emit_c_code(cast(Any, buf))
    c_source = buf.getvalue()

    return RenderedSources(py=py_wrapper, f90=f90_interface, h=c_header, c=c_source)

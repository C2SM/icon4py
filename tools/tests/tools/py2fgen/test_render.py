# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure-string render API.

These tests rely only on string content of the rendered sources; no tempdir,
no CLI, no compile.
"""

from __future__ import annotations

import pytest

from icon4py.tools.py2fgen import _generator, _render


@pytest.fixture
def square_plugin():
    return _generator.get_cffi_description(
        "icon4py.bindings.simple",
        ["square_from_function"],
        "square_plugin",
    )


def test_render_returns_all_four_sources(square_plugin):
    sources = _render.render(square_plugin, h_basename="square_plugin.h")
    assert sources.py
    assert sources.f90
    assert sources.h
    assert sources.c


def test_render_python_wrapper_contains_user_function(square_plugin):
    sources = _render.render(square_plugin, h_basename="square_plugin.h")
    # The Python wrapper drives CFFI's @ffi.def_extern bindings.
    assert "square_from_function" in sources.py
    assert "@ffi.def_extern" in sources.py


def test_render_c_source_embeds_python_wrapper_and_header(square_plugin):
    sources = _render.render(square_plugin, h_basename="custom_header.h")
    # CFFI bakes the Python wrapper into the .c via embedding_init_code.
    assert "square_from_function" in sources.c
    # The #include line must reflect the requested h_basename.
    assert '#include "custom_header.h"' in sources.c


def test_render_c_source_h_basename_threading(square_plugin):
    """Two renders with different h_basenames must produce different .c content."""
    a = _render.render(square_plugin, h_basename="a.h")
    b = _render.render(square_plugin, h_basename="b.h")
    assert '#include "a.h"' in a.c
    assert '#include "b.h"' in b.c
    assert a.c != b.c


def test_render_fortran_module_is_named_after_library(square_plugin):
    import re

    sources = _render.render(square_plugin, h_basename="square_plugin.h")
    # The generated F90 must declare a module whose name matches library_name —
    # check the actual ``module <library_name>`` line, not just any occurrence
    # of the string elsewhere (e.g. inside a bind(c, name=...) attribute).
    assert re.search(r"(?im)^\s*module\s+square_plugin\b", sources.f90)


def test_render_c_header_declares_user_function(square_plugin):
    sources = _render.render(square_plugin, h_basename="square_plugin.h")
    assert "square_from_function" in sources.h


def test_render_is_pure(square_plugin):
    """Rendering twice with identical inputs must be deterministic."""
    a = _render.render(square_plugin, h_basename="square_plugin.h")
    b = _render.render(square_plugin, h_basename="square_plugin.h")
    assert a == b

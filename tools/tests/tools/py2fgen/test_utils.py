# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from icon4py.tools.py2fgen._utils import write_file_if_changed


def test_write_file_if_changed_creates_new_file(tmp_path: pathlib.Path):
    assert write_file_if_changed("hello", tmp_path, "test.txt") is True
    assert (tmp_path / "test.txt").read_text() == "hello"


def test_write_file_if_changed_skips_unchanged(tmp_path: pathlib.Path):
    write_file_if_changed("hello", tmp_path, "test.txt")
    assert write_file_if_changed("hello", tmp_path, "test.txt") is False
    assert (tmp_path / "test.txt").read_text() == "hello"


def test_write_file_if_changed_writes_when_changed(tmp_path: pathlib.Path):
    write_file_if_changed("hello", tmp_path, "test.txt")
    assert write_file_if_changed("world", tmp_path, "test.txt") is True
    assert (tmp_path / "test.txt").read_text() == "world"


def test_write_file_if_changed_force_rewrites_unchanged(tmp_path: pathlib.Path):
    write_file_if_changed("hello", tmp_path, "test.txt")
    assert write_file_if_changed("hello", tmp_path, "test.txt", force=True) is True
    assert (tmp_path / "test.txt").read_text() == "hello"

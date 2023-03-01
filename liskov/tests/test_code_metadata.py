# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from icon4py.liskov.external.exceptions import MissingGitError
from icon4py.liskov.external.metadata import CodeMetadata


@pytest.fixture
def module_parent():
    import icon4py.liskov.external.metadata as meta

    return Path(meta.__file__).parent


def test_generated_on():
    with mock.patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 3, 1, 12, 0, 0)
        metadata = CodeMetadata()
        assert metadata.generated_on == "2023-03-01 12:00:00"


def test_tag(module_parent):
    with mock.patch("subprocess.check_output", side_effect=[b"v1.0.0"]) as mock_git:
        metadata = CodeMetadata()
        assert metadata.tag == "v1.0.0"
        mock_git.assert_called_once_with(
            ["git", "describe", "--tags", "--abbrev=0"], cwd=module_parent
        )


def test_commit_hash(module_parent):
    with mock.patch(
        "subprocess.check_output", side_effect=[b"abcdef123456"]
    ) as mock_git:
        metadata = CodeMetadata()
        assert metadata.commit_hash == "abcdef123456"
        mock_git.assert_any_call(["git", "rev-parse", "HEAD"], cwd=module_parent)


def test_no_git():
    with mock.patch("subprocess.check_output", side_effect=MissingGitError()):
        metadata = CodeMetadata()
        with pytest.raises(MissingGitError):
            metadata.tag


def test_click_context():
    with mock.patch("click.get_current_context") as mock_click:
        mock_click.return_value.params = {"foo": "bar"}
        metadata = CodeMetadata()
        assert metadata.cli_params == {"foo": "bar"}

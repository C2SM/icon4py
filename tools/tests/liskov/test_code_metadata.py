# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

import icon4pytools
from icon4pytools.liskov.external.metadata import CodeMetadata


@pytest.fixture
def module_parent():
    import icon4pytools.liskov.external.metadata as meta

    return Path(meta.__file__).parent


def test_generated_on():
    with mock.patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 3, 1, 12, 0, 0)
        metadata = CodeMetadata()
        assert metadata.generated_on == "2023-03-01 12:00:00"


def test_version(module_parent):
    metadata = CodeMetadata()
    assert metadata.version == icon4pytools.__version__


def test_click_context():
    with mock.patch("click.get_current_context") as mock_click:
        mock_click.return_value.params = {"foo": "bar"}
        metadata = CodeMetadata()
        assert metadata.cli_params == {"foo": "bar"}

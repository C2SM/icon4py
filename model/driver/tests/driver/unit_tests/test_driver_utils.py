# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``driver.driver_utils`` (data-free)."""

import dataclasses
import itertools
import pathlib

import pytest

from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.driver import config as driver_config, driver_utils


_DATA_DIR = pathlib.Path(__file__).parent / "data"


# MCH_CH_R04B09 has divdamp_fac = 0.004, so the enhanced factor is 8 * 0.004 = 0.032.
# That is the value of divdamp_fac_o2 recorded in its solve-nonhydro savepoints.
@pytest.mark.parametrize(
    ("elapsed_time_in_seconds", "expected"),
    [
        (0.0, 0.032),
        (5.0, 0.032),
        (1800.0, 0.032),
        # linear decrease from 8 * divdamp_fac to zero between 1800 s and 7200 s
        (4500.0, 0.016),
        (7200.0, 0.0),
        (7201.0, 0.0),
    ],
)
def test_spinup_second_order_divdamp_factor(
    elapsed_time_in_seconds: float, expected: float
) -> None:
    factor = driver_utils.spinup_second_order_divdamp_factor(
        elapsed_time_in_seconds=elapsed_time_in_seconds,
        fourth_order_divdamp_factor=0.004,
    )
    assert factor == pytest.approx(expected)


def test_spinup_second_order_divdamp_factor_decreases_after_the_initial_period() -> None:
    factors = [
        driver_utils.spinup_second_order_divdamp_factor(
            elapsed_time_in_seconds=float(elapsed_time_in_seconds),
            fourth_order_divdamp_factor=0.004,
        )
        for elapsed_time_in_seconds in range(1800, 7201, 100)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(factors)), factors


def _minimal_experiment_config(with_io: bool = False) -> driver_config.ExperimentConfig:
    config_str = (_DATA_DIR / "minimal_config.yml").read_text()
    if with_io:
        config_str += (_DATA_DIR / "io_section.yml").read_text()
    return config_io.read_yaml_str(config_str, driver_config.ExperimentConfig)


def test_write_experiment_config_without_io_goes_to_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _minimal_experiment_config()
    assert config.io is None

    path = driver_utils.write_experiment_config(
        config, decomposition_defs.SingleNodeProcessProperties()
    )

    assert path == tmp_path / "foo.yaml"
    assert config_io.read_yaml_str(path.read_text(), driver_config.ExperimentConfig) == config


def test_write_experiment_config_with_io_goes_to_output_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "run_output"
    output_path.mkdir()
    config = _minimal_experiment_config(with_io=True)
    assert config.io is not None
    config = config.with_overrides(io=dataclasses.replace(config.io, output_path=output_path))

    path = driver_utils.write_experiment_config(
        config, decomposition_defs.SingleNodeProcessProperties()
    )

    assert path == output_path / "foo.yaml"
    assert not (tmp_path / "foo.yaml").exists()
    assert config_io.read_yaml_str(path.read_text(), driver_config.ExperimentConfig) == config


def test_write_experiment_config_only_on_rank_zero(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    other_rank = decomposition_defs.SingleNodeProcessProperties()
    object.__setattr__(other_rank, "rank", 1)

    assert driver_utils.write_experiment_config(_minimal_experiment_config(), other_rank) is None
    assert list(tmp_path.iterdir()) == []

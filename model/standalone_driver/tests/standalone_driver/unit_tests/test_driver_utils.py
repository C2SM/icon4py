"""Unit tests for ``standalone_driver.driver_utils`` (data-free)."""

import pytest

from icon4py.model.standalone_driver import driver_utils


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
    assert all(later < earlier for earlier, later in zip(factors, factors[1:], strict=False)), (
        factors
    )

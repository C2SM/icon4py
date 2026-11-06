# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, ClassVar, Literal

from gt4py import next as gtx


__all__ = ["Quantity", "Tendency"]


class Quantity:
    """
    A type-level tag, one subclass per quantity at one place on the grid, never instantiated.

    The metadata sits on the class; a keyword not given is inherited from the base (a marker
    base like `Tendency` gives none).
    """

    standard_name: ClassVar[str | None] = None
    units: ClassVar[str]
    long_name: ClassVar[str | None] = None

    # NOTE: temporary, components MWE part 03 (dims): the dims are a property of the field, not of
    # the quantity; they move to the field type (`Field[Q, D, T]`)
    dims: ClassVar[tuple[gtx.Dimension, ...]]
    # NOTE: temporary, components MWE part 03 (dims): the dtype is a property of the field, not of
    # the quantity; they move to the field type (`Field[Q, D, T]`)
    precision: ClassVar[Literal["wp", "vp"]] = "wp"

    def __init_subclass__(
        cls,
        *,
        standard_name: str | None = None,
        units: str | None = None,
        long_name: str | None = None,
        dims: tuple[gtx.Dimension, ...] | None = None,
        precision: Literal["wp", "vp"] | None = None,
    ) -> None:
        super().__init_subclass__()
        given = {
            "standard_name": standard_name,
            "units": units,
            "long_name": long_name,
            "dims": dims,
            "precision": precision,
        }
        for name, value in given.items():
            if value is not None:
                setattr(cls, name, value)

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"'{cls.__name__}' is a type-level tag, not a value.")


class Tendency(Quantity):
    """Marker base of the tendencies."""

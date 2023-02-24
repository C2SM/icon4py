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

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Sequence


class Step(ABC):
    """Abstract base class for pipeline steps.

    Defines the interface for pipeline steps to implement.
    """

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Abstract method to be implemented by concrete steps.

        Args:
            data (Any): Data to be processed.

        Returns:
            Any: Processed data to be passed to the next step.

        """
        pass


class LinearPipelineComposer:
    """Creates a linear pipeline for executing a sequence of functions (steps).

    Args:
        steps (List): List of functions to be executed in the pipeline.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = steps

    def execute(self, data: Any = None) -> Any:
        """Execute all pipeline steps."""
        for step in self.steps:
            data = step(data)
        return data


def linear_pipeline(func: Callable) -> Callable:
    """Apply a linear pipeline to a function using a decorator.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        steps = func(*args, **kwargs)
        composer = LinearPipelineComposer(steps)
        return composer.execute()

    return wrapper

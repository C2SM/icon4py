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

from typing import Callable

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.codegen.serialisation.interface import SerialisationInterface
from icon4py.liskov.common import Step
from icon4py.liskov.logger import setup_logger


logger = setup_logger(__name__)


class InitDataFactory:
    pass


class SavepointDataFactory:
    pass


class SerialisationDeserialiser(Step):
    _FACTORIES: dict[str, Callable] = {
        "Init": InitDataFactory(),
        "Savepoint": SavepointDataFactory(),
    }

    def __call__(self, directives: ts.ParsedDict) -> SerialisationInterface:
        logger.info("Deserialising directives into SerialisationInterface ...")
        deserialised = dict()

        for key, func in self._FACTORIES.items():
            ser = func(directives)
            deserialised[key] = ser

        return SerialisationInterface(**deserialised)

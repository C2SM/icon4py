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

from icon4py.common.logger import setup_logger
from icon4py.liskov.codegen.serialisation.interface import (
    SerialisationCodeInterface,
)
from icon4py.liskov.codegen.shared.deserialiser import Deserialiser


logger = setup_logger(__name__)


class InitDataFactory:
    pass


class SavepointDataFactory:
    pass


class SerialisationCodeDeserialiser(Deserialiser):
    _FACTORIES = {
        "Init": InitDataFactory(),
        "Savepoint": SavepointDataFactory(),
    }
    _INTERFACE_TYPE = SerialisationCodeInterface

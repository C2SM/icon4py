# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, ClassVar, Type

import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.integration.interface import IntegrationCodeInterface
from icon4pytools.liskov.codegen.serialisation.interface import SerialisationCodeInterface
from icon4pytools.liskov.pipeline.definition import Step


logger = setup_logger(__name__)


class Deserialiser(Step):
    _FACTORIES: ClassVar[dict[str, Callable]] = {}
    _INTERFACE_TYPE: ClassVar[Type[SerialisationCodeInterface | IntegrationCodeInterface]]

    def __call__(
        self, directives: ts.ParsedDict
    ) -> SerialisationCodeInterface | IntegrationCodeInterface:
        """Deserialises parsed directives into an Interface object.

        Args:
            directives: A dictionary of parsed directives.

        Returns:
            An Interface object containing the deserialised directives.

        This method is responsible for deserialising parsed directives into an Interface object of the given _INTERFACE_TYPE.
        It uses the `_FACTORIES` dictionary of factory functions to create the appropriate factory object for each directive type.
        The resulting deserialised objects are then used to create an Interface object which can be used for code generation.
        """
        logger.info("Deserialising directives ...")
        deserialised = dict()

        for key, factory in self._FACTORIES.items():
            obj = factory(directives)
            deserialised[key] = obj

        return self._INTERFACE_TYPE(**deserialised)

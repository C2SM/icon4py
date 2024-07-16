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
"""Prognostic one-moment bulk microphysical parameterization.

"""
import sys
from typing import Final
from numpy import sqrt as numpy_sqrt
from numpy import log as numpy_log
from numpy import exp as numpy_exp
import dataclasses
from gt4py.eve.utils import FrozenNamespace
from icon4py.model.common import constants as global_const
from gt4py.next.ffront.decorator import program, field_operator, scan_operator


class DictFrozenNamespace(FrozenNamespace):
    def __init__(self, input_frozen_dict: dict):
        for key, value in input_frozen_dict.items():
            self.key = value


frozen_dict = dict(a=1, b=2, c=3)
# frozen_name = DictFrozenNamespace(frozen_dict)
frozen_name = FrozenNamespace(**frozen_dict)


@dataclasses.dataclass(frozen=True)
class TestData:
    aa: float = 1.0

    def dict1(self):
        _dict = self.__dict__.copy()
        return _dict


class FrozenData(FrozenNamespace):
    aa: float = 1.0


print(frozen_name.a)

test1 = TestData()
print(test1.dict1())
test2 = FrozenNamespace(**vars(test1))

print(test2)

@field_operator
def fxna(
   ztx: float
   ) -> float:
   # Return number of activate ice crystals from temperature
   return ztx + test2.aa
   #return ztx + global_const.ALS

print(fxna(2.0))


class Dummy:
    def __init__(self, input_data: TestData):
        self.frozen_parameters = FrozenNamespace(**vars(input_data))

    def run(self):
        result = self._fxna(5.0)
        return result

    @field_operator
    def _fxna(
        self,
        ztx: float
    ) -> float:
        # Return number of activate ice crystals from temperature
        return ztx + self.frozen_parameters.aa


test_data = TestData(aa=2.0)
dummy = Dummy(test_data)
test_result = dummy.run()
print(test_result)

@dataclasses.dataclass(frozen=True)
class ConstantData:
    aaa: float = 1.0
    bbb: int = 2
    ccc: float = 3.0


constant_data = ConstantData()
hello = {attr: getattr(constant_data, attr) for attr in ["aaa", "bbb"]}
print(hello)

float_attr = [attr for attr in vars(constant_data) if isinstance(getattr(constant_data, attr), float)]
print(float_attr)
# if isinstance(getattr(constant_data, attr), float)
print({attr: getattr(constant_data, attr) for attr in float_attr})


@dataclasses.dataclass(frozen=True)
class ConstantData2:
    aaa2: float = 1.0
    bbb2: int = 2
    ccc2: float = 3.0

constant_data2 = ConstantData2()

test3 = FrozenNamespace(**vars(constant_data),**vars(constant_data2))

print(test3)

# float_attr = [attr for attr in vars(self.config) if isinstance(getattr(self.config, attr), float)]
# constants_in_config = {attr: getattr(self.config, attr) for attr in float_attr}

# self.frozen_empirical_parameters = FrozenNamespace(
#    **constants_in_config, **vars(self.params)
# )

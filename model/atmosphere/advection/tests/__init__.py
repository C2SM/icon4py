# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# On-the-fly building of a (legacy) namespace package for 'tests' using pkgutil
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


class UnsupportedDirectiveError(Exception):
    pass


class DirectiveSyntaxError(Exception):
    pass


class RepeatedDirectiveError(Exception):
    pass


class RequiredDirectivesError(Exception):
    pass


class UnbalancedStencilDirectiveError(Exception):
    pass


class MissingBoundsError(Exception):
    pass


class MissingDirectiveArgumentError(Exception):
    pass

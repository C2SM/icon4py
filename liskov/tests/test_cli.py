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

# todo: use fortran file fixture and test that cli works
import pytest

from icon4py.liskov.cli import main


@pytest.fixture
def simple_f90_file(tmp_f90_file):
    file_content = """\
    !#DSL STENCIL START(test_stencil)
    !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jk = 1, nlev
              DO jc = i_startidx_2, i_endidx_2
                z_rth_pr(jc,jk,1,1) = 0._wp
                z_rth_pr(jc,jk,1,2) = 0._wp
              ENDDO
            ENDDO
    !$ACC END PARALLEL
    !#DSL STENCIL END(test_stencil)
    """
    tmp_f90_file.write_text(file_content)
    return tmp_f90_file


def test_cli(simple_f90_file, cli):
    result = cli.invoke(main, [str(simple_f90_file)])
    assert result.exit_code == 0

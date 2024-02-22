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
import string
import tempfile
from pathlib import Path

import pytest
from pytest import mark

from icon4pytools.liskov.parsing.exceptions import DirectiveSyntaxError
from icon4pytools.liskov.parsing.scan import DirectivesScanner
from icon4pytools.liskov.parsing.types import RawDirective

from .fortran_samples import DIRECTIVES_SAMPLE, NO_DIRECTIVES_STENCIL, SINGLE_FUSED


ALLOWED_EOL_CHARS = [")", "&"]


def scan_tempfile(string: str):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(string.encode())
        tmp.flush()
        scanner = DirectivesScanner(Path(tmp.name))
        return scanner()


def special_char():
    def special_chars_generator():
        for char in string.punctuation:
            yield char

    return special_chars_generator()


@mark.parametrize(
    "string,expected",
    [
        (NO_DIRECTIVES_STENCIL, []),
        (
            DIRECTIVES_SAMPLE,
            [
                RawDirective("!$DSL IMPORTS()\n", 0, 0),
                RawDirective("!$DSL START CREATE()\n", 2, 2),
                RawDirective("!$DSL DECLARE(vn=p_patch%vn; vn2=p_patch%vn2)\n", 4, 4),
                RawDirective(
                    "!$DSL START STENCIL(name=mo_nh_diffusion_06; vn=p_patch%vn; &\n!$DSL       a=a; b=c)\n",
                    6,
                    7,
                ),
                RawDirective("!$DSL END STENCIL(name=mo_nh_diffusion_06)\n", 9, 9),
                RawDirective(
                    "!$DSL START STENCIL(name=mo_nh_diffusion_07; xn=p_patch%xn)\n",
                    11,
                    11,
                ),
                RawDirective("!$DSL END STENCIL(name=mo_nh_diffusion_07)\n", 13, 13),
                RawDirective("!$DSL UNKNOWN_DIRECTIVE()\n", 15, 15),
                RawDirective("!$DSL END CREATE()\n", 16, 16),
            ],
        ),
        (
            SINGLE_FUSED,
            [
                RawDirective(string="    !$DSL IMPORTS()\n", startln=0, endln=0),
                RawDirective(
                    string="    !$DSL INSERT(INTEGER :: start_interior_idx_c, end_interior_idx_c, start_nudging_idx_c, end_halo_1_idx_c)\n",
                    startln=2,
                    endln=2,
                ),
                RawDirective(
                    string="    !$DSL DECLARE(kh_smag_e=nproma,p_patch%nlev,p_patch%nblks_e; &\n    !$DSL      kh_smag_ec=nproma,p_patch%nlev,p_patch%nblks_e; &\n    !$DSL      z_nabla2_e=nproma,p_patch%nlev,p_patch%nblks_e; &\n    !$DSL      kh_c=nproma,p_patch%nlev; &\n    !$DSL      div=nproma,p_patch%nlev; &\n    !$DSL      div_ic=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      hdef_ic=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      z_nabla4_e2=nproma,p_patch%nlev; &\n    !$DSL      vn=nproma,p_patch%nlev,p_patch%nblks_e; &\n    !$DSL      z_nabla2_c=nproma,p_patch%nlev,p_patch%nblks_e; &\n    !$DSL      dwdx=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      dwdy=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      w=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      enh_diffu_3d=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      z_temp=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      theta_v=nproma,p_patch%nlev,p_patch%nblks_c; &\n    !$DSL      exner=nproma,p_patch%nlev,p_patch%nblks_c)\n",
                    startln=4,
                    endln=20,
                ),
                RawDirective(
                    string="        !$DSL START FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence; &\n        !$DSL  kh_smag_ec=kh_smag_ec(:,:,1); vn=p_nh_prog%vn(:,:,1); e_bln_c_s=p_int%e_bln_c_s(:,:,1); &\n        !$DSL  geofac_div=p_int%geofac_div(:,:,1); diff_multfac_smag=diff_multfac_smag(:); &\n        !$DSL  wgtfac_c=p_nh_metrics%wgtfac_c(:,:,1); div_ic=p_nh_diag%div_ic(:,:,1); &\n        !$DSL  hdef_ic=p_nh_diag%hdef_ic(:,:,1); &\n        !$DSL  div_ic_abs_tol=1e-18_wp; vertical_lower=2; &\n        !$DSL  vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)\n",
                    startln=22,
                    endln=28,
                ),
                RawDirective(
                    string="        !$DSL START STENCIL(name=temporary_fields_for_turbulence_diagnostics; kh_smag_ec=kh_smag_ec(:,:,1); vn=p_nh_prog%vn(:,:,1); &\n        !$DSL       e_bln_c_s=p_int%e_bln_c_s(:,:,1); geofac_div=p_int%geofac_div(:,:,1); &\n        !$DSL       diff_multfac_smag=diff_multfac_smag(:); kh_c=kh_c(:,:); div=div(:,:); &\n        !$DSL       vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; &\n        !$DSL       horizontal_upper=i_endidx)\n",
                    startln=30,
                    endln=34,
                ),
                RawDirective(
                    string="        !$DSL END STENCIL(name=temporary_fields_for_turbulence_diagnostics)\n",
                    startln=36,
                    endln=36,
                ),
                RawDirective(
                    string="        !$DSL START STENCIL(name=calculate_diagnostics_for_turbulence; div=div; kh_c=kh_c; wgtfac_c=p_nh_metrics%wgtfac_c(:,:,1); &\n        !$DSL               div_ic=p_nh_diag%div_ic(:,:,1); hdef_ic=p_nh_diag%hdef_ic(:,:,1); div_ic_abs_tol=1e-18_wp; &\n        !$DSL               vertical_lower=2; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)\n",
                    startln=38,
                    endln=40,
                ),
                RawDirective(
                    string="        !$DSL END STENCIL(name=calculate_diagnostics_for_turbulence)\n",
                    startln=42,
                    endln=42,
                ),
                RawDirective(
                    string="        !$DSL END FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence)\n",
                    startln=44,
                    endln=44,
                ),
            ],
        ),
    ],
)
def test_directives_scanning(string, expected):
    scanned = scan_tempfile(string)
    assert scanned == expected


@pytest.mark.parametrize("special_char", special_char())
def test_directive_eol(special_char):
    if special_char in ALLOWED_EOL_CHARS:
        pytest.skip()
    else:
        directive = "!$DSL IMPORT(" + special_char
        with pytest.raises(DirectiveSyntaxError):
            scan_tempfile(directive)


def test_directive_unclosed():
    directive = "!$DSL IMPORT(&\n!CALL foo()"
    with pytest.raises(DirectiveSyntaxError):
        scan_tempfile(directive)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatest of the prescribed surface-flux provider (``isrfc_type = 1``).

The serialized idealized experiment (exp.exclaim_ape_aesPhys) runs the tmx
surface scheme in its fixed-surface-flux branch, so ICON's surface fluxes are
an analytic function of the surface pressure and the prescribed SST. This test
verifies :class:`PrescribedFluxProvider` against the ``tmx-surface-fluxes``
savepoint, i.e. against the fluxes the Fortran actually fed into the
atmospheric diffusion.

That matters beyond the granule tests, which read those fluxes from the
savepoint: a driver run has no savepoint to read from and has to reproduce
them, and feeding zeros instead leaves the lowest model level without its
~-83 W/m^2 of sensible heat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import surface_fluxes, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.config import SurfaceType, TmxConfig
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from ..fixtures import load_fortran_dict
from .utils import TMX_DATES


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


# The provider reproduces the Fortran to a couple of ulp; this is the largest
# relative deviation measured on the v08 archive with ~20x of headroom.
RTOL: float = 1.0e-14


def ape_surface_temperature(input_dict: dict[str, Any]) -> float:
    """The globally constant aquaplanet SST [K] of the testcase namelist.

    Port of 'ape_sst_const' (mo_ape_params.f90:173-183): ``tmelt +
    ape_sst_val``. Only the constant-SST case is supported; the latitude
    dependent 'sst1'..'sst_qobs' profiles would need a per-cell field.
    """
    testcase = input_dict["nh_testcase_nml"]
    sst_case = testcase["ape_sst_case"]
    if sst_case != "sst_const":
        raise NotImplementedError(f"Only 'sst_const' is supported, got '{sst_case}'.")
    return constants.MELTING_TEMPERATURE + float(testcase["ape_sst_val"])


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_prescribed_surface_fluxes_match_fortran(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    allocator = model_backends.get_allocator(backend)
    input_dict = load_fortran_dict(
        experiment_description=experiment_description,
        process_props=process_props,
        fname=fortran_config.INPUT_DICT_FNAME,
    )
    atm_dict = load_fortran_dict(
        experiment_description=experiment_description,
        process_props=process_props,
        fname=fortran_config.ATM_DICT_FNAME,
    )
    config = TmxConfig.from_fortran_dict(atm_dict=atm_dict, input_dict=input_dict)
    # the archive must be the fixed-flux case, otherwise this test is vacuous
    assert config.surface_type is SurfaceType.FIXED_HEAT_FLUXES

    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    reference = data_provider.from_savepoint_tmx_surface_fluxes(date=date)

    provider = surface_fluxes.PrescribedFluxProvider(
        config=config,
        pressure_ifc=entry_savepoint.pres_ifc(),
        surface_temperature=data_alloc.constant_field(
            icon_grid, ape_surface_temperature(input_dict), dims.CellDim, allocator=allocator
        ),
    )
    out = tmx_states.TmxSurfaceFluxState.allocate(icon_grid, allocator=allocator)
    provider.compute(out=out)

    fields = (
        (out.sensible_heat_flux, reference.hfss(), "hfss"),
        (out.evapotranspiration, reference.evspsbl(), "evspsbl"),
        (out.u_stress, reference.tauu(), "tauu"),
        (out.v_stress, reference.tauv(), "tauv"),
        (out.q_snocpymlt, reference.q_snocpymlt(), "q_snocpymlt"),
    )
    for actual, desired, name in fields:
        test_utils.assert_dallclose(
            actual.asnumpy(), desired.asnumpy(), rtol=RTOL, atol=0.0, err_msg=name
        )

    # guard against a vacuous pass: the sensible heat flux is the one field
    # that is not identically zero in this configuration
    assert abs(out.sensible_heat_flux.asnumpy()).min() > 0.0


def test_surface_flux_config_defaults_match_fortran() -> None:
    """The defaults mirror the initialization in mo_nh_testcases_nml.f90:280-282."""
    config = TmxConfig()
    assert config.surface_type is SurfaceType.INTERACTIVE
    assert config.shflx == 0.1
    assert config.lhflx == 0.0


def test_prescribed_flux_provider_rejects_interactive_surface() -> None:
    """The interactive surface scheme is not ported."""
    grid = simple.simple_grid()
    with pytest.raises(NotImplementedError, match="FIXED_HEAT_FLUXES"):
        surface_fluxes.PrescribedFluxProvider(
            config=TmxConfig(surface_type=SurfaceType.INTERACTIVE),
            pressure_ifc=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
            ),
            surface_temperature=data_alloc.constant_field(grid, 300.0, dims.CellDim),
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing

from icon4py.model.common.config import config_io
from icon4py.model.common.initial_condition import config as ic_config
from icon4py.model.common.topography import config as topo_config


@dataclasses.dataclass
class DomainConfig:
    initial_condition: ic_config.IC_CONFIG
    topography: topo_config.TOPO_CONFIG


@config_io.CONV.register_structure_hook
def structure_domain(spec: dict, _: typing.Any) -> DomainConfig:
    params = spec.pop("params", {})

    ic_structurer = typing.cast(
        config_io.ConfigUnionStructurer,
        config_io.CONV.get_structure_hook(ic_config.IC_CONFIG.__value__),
    )
    ic_type = ic_structurer.type_map[spec["initial_condition"]["type"]]
    ic_params = {f.name for f in dataclasses.fields(ic_type)}

    topo_structurer = typing.cast(
        config_io.ConfigUnionStructurer,
        config_io.CONV.get_structure_hook(topo_config.TOPO_CONFIG.__value__),
    )
    topo_type = topo_structurer.type_map[spec["topography"]["type"]]
    topo_params = {f.name for f in dataclasses.fields(topo_type)}

    requested_params = ic_params | topo_params
    if extra_params := set(params.keys()) - requested_params:
        raise TypeError(
            f"Extra parameters found, not used by either initial condition type '{spec['initial_condition']['type']}' or topography type '{spec['topography']['type']}': {sorted(extra_params)}"
        )

    ic = ic_type(**{k: v for k, v in params.items() if k in ic_params})
    topo = topo_type(**{k: v for k, v in params.items() if k in topo_params})
    return DomainConfig(initial_condition=ic, topography=topo)


@config_io.CONV.register_unstructure_hook
def unstructure_domain(domain: DomainConfig) -> dict:
    ic_spec = config_io.CONV.unstructure(
        domain.initial_condition, unstructure_as=ic_config.IC_CONFIG.__value__
    )
    topo_spec = config_io.CONV.unstructure(
        domain.topography, unstructure_as=topo_config.TOPO_CONFIG.__value__
    )
    ic_type = {"type": ic_spec.pop("type")}
    topo_type = {"type": topo_spec.pop("type")}
    params = ic_spec | topo_spec
    return {
        "params": params,
        "initial_condition": ic_type,
        "topography": topo_type,
    }

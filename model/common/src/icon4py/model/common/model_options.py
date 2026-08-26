# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import dace
import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common import backend_configuration as backend_cfg, dimension, model_backends


log = logging.getLogger(__name__)


def dict_values_to_list(d: dict[str, Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def _dace_remove_access_node_copies(sdfg: dace.SDFG) -> None:
    sdfg.apply_transformations_repeated(
        gtx_transformations.RemoveAccessNodeCopies(),
        validate=False,
        validate_all=False,
    )


def _set_program_specific_dace_options(
    program_name: str,
    device: model_backends.DeviceType | None,
    backend_descriptor: model_backends.BackendDescriptor,
    optimization_args: dict[str, Any],
    optimization_hooks: dict[Any, Any],
) -> None:
    if program_name in (
        "vertically_implicit_solver_at_corrector_step",
        "vertically_implicit_solver_at_predictor_step",
    ):
        # Enable pass that removes access node (next_w) copies for vertically implicit solver programs
        optimization_hooks.setdefault(
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep,
            _dace_remove_access_node_copies,
        )
        optimization_args.setdefault("scan_loop_unrolling", True)
        optimization_args.setdefault("scan_loop_unrolling_factor", 0)
    # TODO(havogt): Eventually the option `use_zero_origin` should be removed and the default behavior should be `use_zero_origin=False`.
    # We keep it `True` for 'compute_rho_theta_pgrad_and_update_vn' as performance drops,
    # due to it falling into a less optimized code generation (on santis).
    if program_name == "compute_rho_theta_pgrad_and_update_vn":
        backend_descriptor["use_zero_origin"] = True
    if program_name == "graupel_run":
        optimization_args["fuse_tasklets"] = True
        if device != model_backends.DeviceType.ROCM:
            optimization_args["gpu_maxnreg"] = 80
            optimization_args["gpu_block_size_2d"] = (64, 6)
        optimization_args["gpu_memory_pool"] = False
        optimization_args["make_persistent"] = True


def _set_device_specific_dace_options(
    device: model_backends.DeviceType | None,
    has_external_workspace: bool,
    optimization_args: dict[str, Any],
) -> None:
    if device == model_backends.DeviceType.ROCM:
        if not has_external_workspace:
            # Only needed when no external workspace is provided (i.e.
            # ICON4PY_BACKEND_WORKSPACE_SIZE is not set); the external
            # workspace already avoids the runtime allocation overhead.
            optimization_args["gpu_memory_pool"] = False
            optimization_args["make_persistent"] = True
        # AMD MI300A: (256,1,1) for 2D maps gives ~20% speedup on the solver.
        # All threads on Cell dimension maximizes coalescing on MI300A.
        optimization_args.setdefault("gpu_block_size_2d", (256, 1, 1))
        # Setting a block size of (256,1,1) for 1D maps doesn't give a significant
        # speedup on MI300A but it doesn't hurt either
        optimization_args.setdefault("gpu_block_size_1d", (256, 1, 1))
        # Vertical blocking with length 4 is adding ~12% speedup on MI300A for dycore
        optimization_args["blocking_dims"] = list(dimension.vertical_dims())
        optimization_args["blocking_size"] = 4
        optimization_args["blocking_only_if_independent_nodes"] = False
        if os.getenv("CUPY_ACCELERATORS") == "":
            log.warning(
                'CUPY_ACCELERATORS environment variable should be set to "cub" otherwise reductions have degraded performance on AMD GPUs.'
            )
    elif device == model_backends.DeviceType.CUDA:
        optimization_args.setdefault("gpu_block_size_2d", (128, 2, 1))
        optimization_args.setdefault("gpu_block_size_2d", (256, 1, 1))


def get_dace_options(
    program_name: str,
    backend_config: backend_cfg.BackendConfig | None,
    **backend_descriptor: Any,
) -> model_backends.BackendDescriptor:
    device = backend_descriptor.get("device")
    optimization_args = backend_descriptor.get("optimization_args", {})
    optimization_hooks = optimization_args.get("optimization_hooks", {})

    if backend_config is not None:
        # The workspace memory allows to avoid the overhead of runtime allocations,
        # which are expensive in the AMD runtime.
        backend_descriptor["external_workspace"] = backend_cfg.ICON_WORKSPACE_ALLOCATOR.allocate(
            device,
            size=backend_config.workspace_size,
            alignment=backend_config.workspace_alignment,
        )
        optimization_args["transient_memory_mode"] = (
            gtx_transformations.TransientMemoryMode.EXTERNAL
        )

    _set_program_specific_dace_options(
        program_name, device, backend_descriptor, optimization_args, optimization_hooks
    )
    _set_device_specific_dace_options(
        device,
        has_external_workspace=backend_config is not None,
        optimization_args=optimization_args,
    )

    if optimization_hooks:
        optimization_args["optimization_hooks"] = optimization_hooks
    if optimization_args:
        backend_descriptor["optimization_args"] = optimization_args
    return backend_descriptor


def get_gtfn_options(
    program_name: str, **backend_descriptor: Any
) -> model_backends.BackendDescriptor:
    return backend_descriptor


def get_options(
    program_name: str,
    *,
    backend_config: backend_cfg.BackendConfig | None,
    **backend_descriptor: Any,
) -> model_backends.BackendDescriptor:
    if "backend_factory" not in backend_descriptor:
        # here we could set a backend_factory per program
        backend_descriptor["backend_factory"] = model_backends.make_custom_dace_backend
    if backend_descriptor["backend_factory"] == model_backends.make_custom_dace_backend:
        backend_descriptor = get_dace_options(program_name, backend_config, **backend_descriptor)
    if backend_descriptor["backend_factory"] == model_backends.make_custom_gtfn_backend:
        backend_descriptor = get_gtfn_options(program_name, **backend_descriptor)

    return backend_descriptor


def customize_backend(
    program: gtx_typing.Program | gtx.typing.FieldOperator | None,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    backend_config: backend_cfg.BackendConfig | None = None,
) -> gtx_typing.Backend | None:
    backend_config = backend_config or backend_cfg.backend_config_from_env()
    program_name = program.__name__ if program is not None else ""
    if backend is None or isinstance(backend, gtx_backend.Backend):
        backend_name = backend.name if backend is not None else "embedded"
        log.info(f"Using non-custom backend '{backend_name}' for '{program_name}'.")
        return backend  # type: ignore[return-value]

    backend_descriptor = (
        {"device": backend} if isinstance(backend, model_backends.DeviceType) else backend
    )
    backend_descriptor = get_options(
        program_name, backend_config=backend_config, **backend_descriptor
    )
    backend_descriptor["device"] = backend_descriptor.get(
        "device", model_backends.CPU
    )  # set default device
    backend_factory = backend_descriptor.pop(
        "backend_factory", model_backends.make_custom_dace_backend
    )
    custom_backend = backend_factory(**backend_descriptor)
    log.info(
        f"Using custom backend '{custom_backend.name}' for '{program_name}' with options: {backend_descriptor}."
    )
    return custom_backend


def setup_program(
    *,
    program: gtx_typing.Program,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    constant_args: dict[str, gtx.Field | gtx_typing.Scalar] | None = None,
    variants: dict[str, list[gtx_typing.Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: gtx_typing.OffsetProvider | None = None,
    backend_config: backend_cfg.BackendConfig | None = None,
) -> Callable[..., None]:
    """
    This function processes arguments to the GT4Py program. It
    - binds arguments that don't change during model run ('constant_args', 'horizontal_sizes', 'vertical_sizes');
    - inlines scalar arguments into the GT4Py program at compile-time (via GT4Py's 'compile').
    Args:
        - backend: GT4Py backend,
        - program: GT4Py program,
        - constant_args: constant fields and scalars,
        - variants: list of all scalars potential values from which one is selected at run time,
        - horizontal_sizes: horizontal domain bounds,
        - vertical_sizes: vertical domain bounds,
        - offset_provider: GT4Py offset_provider,
        - backend_config: external DaCe workspace sizing, or `None` to fall back
          to the 'ICON4PY_BACKEND_WORKSPACE_<SIZE|ALIGNMENT>' environment variables.
    """
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    backend = customize_backend(program, backend, backend_config=backend_config)

    bound_static_args = {k: v for k, v in constant_args.items() if gtx.is_scalar_type(v)}
    static_args_program = program.with_backend(backend)
    if backend is not None:
        static_args_program = static_args_program.with_compilation_options(enable_jit=False)
        static_args_program.compile(
            **dict_values_to_list(horizontal_sizes),
            **dict_values_to_list(vertical_sizes),
            **variants,
            **dict_values_to_list(bound_static_args),
            offset_provider=offset_provider,
        )

    return functools.partial(
        static_args_program,
        **constant_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )

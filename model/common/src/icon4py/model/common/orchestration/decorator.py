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

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from gt4py.next import CompileTimeConnectivity
from gt4py.next.common import Connectivity

from icon4py.model.common import settings
from icon4py.model.common.decomposition.definitions import SingleNodeResult
from icon4py.model.common.decomposition.mpi_decomposition import MultiNodeResult


try:
    import dace
except ImportError:
    from types import ModuleType

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


def dace_orchestration() -> bool:
    """DaCe Orchestration: GT4Py Programs called in a dace.program annotated function"""
    if settings.dace_orchestration is not None:
        return True
    return False


def orchestration(method=True):
    def decorator(fuse_func):
        compiled_sdfgs = {}  # Caching

        def wrapper(*args, **kwargs):
            if dace_orchestration():
                if "dace" not in settings.backend.executor.name.lower():
                    raise ValueError(
                        "DaCe Orchestration works only with DaCe backends. Change the backend to a DaCe supported one."
                    )

                if method:
                    # self is used to retrieve the _exchange object -on the fly halo exchanges- and the grid object -offset providers-
                    self = args[0]
                    self_name = next(
                        param.name for param in inspect.signature(fuse_func).parameters.values()
                    )
                else:
                    raise ValueError(
                        "The orchestration decorator is only for methods -at least for now-."
                    )

                has_exchange = False
                has_grid = False
                for attr_name, attr_value in self.__dict__.items():
                    if (
                        isinstance(attr_value, GHexMultiNodeExchange)
                        or isinstance(attr_value, SingleNodeExchange)
                        and not has_exchange
                    ):
                        has_exchange = True
                        exchange_obj = getattr(self, attr_name)
                    if isinstance(attr_value, IconGrid) and not has_grid:
                        has_grid = True
                        grid = getattr(self, attr_name)

                if not has_exchange or not has_grid:
                    raise ValueError("No exchnage/grid object found.")

                compile_time_args_kwargs = {}
                all_args_kwargs = list(args) + list(kwargs.values())
                for i, (k, v) in enumerate(fuse_func.__annotations__.items()):
                    if v is dace.compiletime:
                        compile_time_args_kwargs[k] = all_args_kwargs[i]
                unique_id = hash(tuple([id(e) for e in compile_time_args_kwargs.values()]))

                default_build_folder = os.path.join(
                    ".dacecache", f"MPI_rank_{exchange_obj.my_rank()}"
                )

                parse_compile_cache_sdfg(
                    unique_id,
                    compiled_sdfgs,
                    default_build_folder,
                    exchange_obj,
                    fuse_func,
                    compile_time_args_kwargs,
                    self_name,
                    simplify_fused_sdfg=True,
                )
                dace_program = compiled_sdfgs[unique_id]["dace_program"]
                sdfg = compiled_sdfgs[unique_id]["sdfg"]
                compiled_sdfg = compiled_sdfgs[unique_id]["compiled_sdfg"]

                with dace.config.temporary_config():
                    configure_dace_temp_env(default_build_folder)

                    updated_args = mod_args_for_dace_structures(fuse_func, args)
                    updated_kwargs = {
                        **kwargs,
                        **dace_specific_kwargs(exchange_obj, grid.offset_providers),
                    }
                    updated_kwargs = {
                        **updated_kwargs,
                        **dace_symbols_concretization(grid, fuse_func, args, kwargs),
                    }

                    sdfg_args = dace_program._create_sdfg_args(sdfg, updated_args, updated_kwargs)
                    if method:
                        del sdfg_args[self_name]

                    return compiled_sdfg(**sdfg_args)
            else:
                return fuse_func(*args, **kwargs)

        return wrapper

    return decorator


def dace_inhibitor(f: Callable):
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return f


@dace_inhibitor
def wait(comm_handle: Union[int, SingleNodeResult, MultiNodeResult]):
    if isinstance(comm_handle, int):
        pass
    else:
        comm_handle.wait()


def build_compile_time_connectivities(
    offset_providers: dict[str, Connectivity],
) -> dict[str, Connectivity]:
    connectivities = {}
    for k, v in offset_providers.items():
        if hasattr(v, "table"):
            connectivities[k] = CompileTimeConnectivity(
                v.max_neighbors, v.has_skip_values, v.origin_axis, v.neighbor_axis, v.table.dtype
            )
        else:
            connectivities[k] = v

    return connectivities


if dace:
    import inspect
    import os
    import shutil
    from typing import Any, Optional

    import dace
    import numpy as np
    from dace import hooks
    from dace.transformation.passes.simplify import SimplifyPass
    from ghex import expose_cpp_ptr
    from gt4py._core import definitions as core_defs
    from gt4py.next.program_processors.runners.dace_iterator.utility import (
        connectivity_identifier,
        field_size_symbol_name,
        field_stride_symbol_name,
    )

    from icon4py.model.common.decomposition.definitions import SingleNodeExchange
    from icon4py.model.common.decomposition.mpi_decomposition import GHexMultiNodeExchange
    from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim
    from icon4py.model.common.grid.icon import IconGrid
    from icon4py.model.common.orchestration.dtypes import *  # noqa: F403

    def dev_type_from_gt4py_to_dace(device_type: core_defs.DeviceType) -> dace.dtypes.DeviceType:
        if device_type == core_defs.DeviceType.CPU:
            return dace.dtypes.DeviceType.CPU
        elif device_type == core_defs.DeviceType.CUDA:
            return dace.dtypes.DeviceType.GPU
        else:
            raise ValueError("The device type is not supported.")

    def parse_compile_cache_sdfg(
        unique_id: int,
        compiled_sdfgs: dict[int, dace.codegen.compiled_sdfg.CompiledSDFG],
        default_build_folder: str,
        exchange_obj: Union[SingleNodeExchange, GHexMultiNodeExchange],
        fuse_func: Callable,
        compile_time_args_kwargs: dict[str, Any],
        self_name: Optional[str] = None,
        simplify_fused_sdfg: bool = True,
    ) -> None:
        """Function that parses, compiles and caches the fused SDFG along with adding the halo exchanges."""
        if unique_id in compiled_sdfgs:
            return

        compiled_sdfgs[unique_id] = {}

        with dace.config.temporary_config():
            device_type = configure_dace_temp_env(default_build_folder)

            compiled_sdfgs[unique_id]["dace_program"] = dace.program(
                auto_optimize=False,
                device=dev_type_from_gt4py_to_dace(device_type),
                distributed_compilation=False,
            )(fuse_func)
            dace_program = compiled_sdfgs[unique_id]["dace_program"]

            cache_sanitization(default_build_folder, exchange_obj)

            cache_from_disk = get_env_bool("DACE_ORCH_CACHE_FROM_DISK", default=False)
            dace_program_location = os.path.join(
                default_build_folder,
                f"{fuse_func.__module__}.{fuse_func.__name__}".replace(".", "_"),
            )

            if cache_from_disk and os.path.exists(
                os.path.join(dace_program_location, "program.sdfg")
            ):
                try:
                    if self_name:
                        self = compile_time_args_kwargs.pop(self_name)
                        compiled_sdfgs[unique_id]["sdfg"], _ = dace_program.load_sdfg(
                            os.path.join(dace_program_location, "program.sdfg"),
                            self,
                            **compile_time_args_kwargs,
                        )
                        (
                            compiled_sdfgs[unique_id]["compiled_sdfg"],
                            _,
                        ) = dace_program.load_precompiled_sdfg(
                            dace_program_location, self, **compile_time_args_kwargs
                        )
                    else:
                        compiled_sdfgs[unique_id]["sdfg"], _ = dace_program.load_sdfg(
                            os.path.join(dace_program_location, "program.sdfg"),
                            **compile_time_args_kwargs,
                        )
                        (
                            compiled_sdfgs[unique_id]["compiled_sdfg"],
                            _,
                        ) = dace_program.load_precompiled_sdfg(
                            dace_program_location, **compile_time_args_kwargs
                        )
                except:  # noqa: E722
                    raise ValueError(
                        "Corrupted cache. Remove `.dacecache` folder and re-run the program."
                    ) from None
            else:
                if self_name:
                    self = compile_time_args_kwargs.pop(self_name)
                    compiled_sdfgs[unique_id]["sdfg"] = dace_program.to_sdfg(
                        self, **compile_time_args_kwargs, simplify=False, validate=True
                    )
                else:
                    compiled_sdfgs[unique_id]["sdfg"] = dace_program.to_sdfg(
                        **compile_time_args_kwargs, simplify=False, validate=True
                    )
                sdfg = compiled_sdfgs[unique_id]["sdfg"]

                if sdfg.name != f"{fuse_func.__module__}.{fuse_func.__name__}".replace(".", "_"):
                    raise ValueError(
                        "fused_SDFG.name != {fuse_func.__module__}.{fuse_func.__name__}"
                    )

                if simplify_fused_sdfg:
                    SimplifyPass(
                        validate=True,
                        validate_all=False,
                        verbose=False,
                        skip={
                            "InlineSDFGs",
                            "DeadDataflowElimination",
                        },
                    ).apply_pass(sdfg, {})
                    # Alternatively:
                    # sdfg.simplify(validate=True) # noqa: ERA001

                exchange_obj.num_of_halo_tasklets = 0  # reset the counter for the next fused SDFG

                sdfg.save(os.path.join(dace_program_location, "program.sdfg"))

                with hooks.invoke_sdfg_call_hooks(sdfg) as sdfg_:
                    # TODO(kotsaloscv): Re-think the distributed compilation -all args need to be type annotated and not dace.compiletime-
                    compiled_sdfgs[unique_id]["compiled_sdfg"] = sdfg_.compile(
                        validate=dace_program.validate
                    )

    def get_env_bool(env_var_name: str, default: bool = False) -> bool:
        value = os.getenv(env_var_name, str(default)).lower()
        return value in ("true", "1")

    def count_folders_in_directory(directory: str) -> int:
        try:
            entries = os.listdir(directory)
            folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
            return len(folders)
        except Exception as e:
            return str(e)

    def cache_sanitization(
        default_build_folder: str, exchange_obj: Union[SingleNodeExchange, GHexMultiNodeExchange]
    ) -> None:
        normalized_path = os.path.normpath(default_build_folder)
        parts = normalized_path.split(os.sep)
        dacecache = next(part for part in parts if part)  # normally .dacecache
        if count_folders_in_directory(dacecache) != exchange_obj.get_size():
            try:
                shutil.rmtree(dacecache)
            except Exception as e:
                print(f"Error: {e}")

    def configure_dace_temp_env(default_build_folder: str) -> core_defs.DeviceType:
        dace.config.Config.set("default_build_folder", value=default_build_folder)
        dace.config.Config.set(
            "compiler", "allow_view_arguments", value=True
        )  # Allow numpy views as arguments: If true, allows users to call DaCe programs with NumPy views (for example, “A[:,1]” or “w.T”)
        dace.config.Config.set(
            "optimizer", "automatic_simplification", value=False
        )  # simplifications & optimizations after placing halo exchanges -need a sequential structure of nested sdfgs-
        dace.config.Config.set("optimizer", "autooptimize", value=False)
        device_type = settings.backend.executor.otf_workflow.step.translation.device_type
        if device_type == core_defs.DeviceType.CPU:
            device = "cpu"
            compiler_args = dace.config.Config.get("compiler", "cpu", "args")
            compiler_args = compiler_args.replace("-std=c++14", "-std=c++17")  # needed for GHEX
            # disable finite-math-only in order to support isfinite/isinf/isnan builtins
            if "-ffast-math" in compiler_args:
                compiler_args += " -fno-finite-math-only"
            if "-ffinite-math-only" in compiler_args:
                compiler_args = compiler_args.replace("-ffinite-math-only", "")
        elif device_type == core_defs.DeviceType.CUDA:
            device = "cuda"
            compiler_args = dace.config.Config.get("compiler", "cuda", "args")
            compiler_args = compiler_args.replace("-Xcompiler", "-Xcompiler -std=c++17")
            compiler_args += " -std=c++17"
        else:
            raise ValueError("The device type is not supported.")
        dace.config.Config.set("compiler", device, "args", value=compiler_args)

        return device_type

    def get_stride_from_numpy_to_dace(numpy_array: np.ndarray, axis: int) -> int:
        """
        GHEX/NumPy strides: number of bytes to jump
        DaCe strides: number of elements to jump
        """
        return numpy_array.strides[axis] // numpy_array.itemsize

    def dace_specific_kwargs(
        exchange_obj: Union[SingleNodeExchange, GHexMultiNodeExchange],
        offset_providers: dict[str, Connectivity],
    ) -> dict[str, Any]:
        return {
            # connectivity tables
            **{
                connectivity_identifier(k): v.table
                for k, v in offset_providers.items()
                if hasattr(v, "table")
            },
            # GHEX C++ ptrs
            "__context_ptr": expose_cpp_ptr(exchange_obj._context)
            if isinstance(exchange_obj, GHexMultiNodeExchange)
            else 0,
            "__comm_ptr": expose_cpp_ptr(exchange_obj._comm)
            if isinstance(exchange_obj, GHexMultiNodeExchange)
            else 0,
            **{
                f"__pattern_{dim.value}Dim_ptr": expose_cpp_ptr(exchange_obj._patterns[dim])
                if isinstance(exchange_obj, GHexMultiNodeExchange)
                else 0
                for dim in (CellDim, VertexDim, EdgeDim)
            },
            **{
                f"__domain_descriptor_{dim.value}Dim_ptr": expose_cpp_ptr(
                    exchange_obj._domain_descriptors[dim].__wrapped__
                )
                if isinstance(exchange_obj, GHexMultiNodeExchange)
                else 0
                for dim in (CellDim, VertexDim, EdgeDim)
            },
        }

    def dace_symbols_concretization(
        grid: IconGrid, fuse_func: Callable, args: Any, kwargs: Any
    ) -> dict[str, Any]:
        flattened_xargs_type_value = list(
            zip(
                list(fuse_func.__annotations__.values()),
                list(args) + list(kwargs.values()),
                strict=False,
            )
        )
        return {
            **{
                "CellDim_sym": grid.offset_providers["C2E"].table.shape[0],
                "EdgeDim_sym": grid.offset_providers["E2C"].table.shape[0],
                "KDim_sym": grid.num_levels,
            },
            **{
                f"DiffusionDiagnosticState_{member}_s{stride!s}_sym": get_stride_from_numpy_to_dace(
                    getattr(k_v[1], member).ndarray, stride
                )
                for k_v in flattened_xargs_type_value
                for member in ["hdef_ic", "div_ic", "dwdx", "dwdy"]
                for stride in [0, 1]
                if k_v[0] is DiffusionDiagnosticState_t  # noqa: F405
            },
            **{
                f"PrognosticState_{member}_s{stride!s}_sym": get_stride_from_numpy_to_dace(
                    getattr(k_v[1], member).ndarray, stride
                )
                for k_v in flattened_xargs_type_value
                for member in ["rho", "w", "vn", "exner", "theta_v"]
                for stride in [0, 1]
                if k_v[0] is PrognosticState_t  # noqa: F405
            },
            **{
                field_size_symbol_name(connectivity_identifier(k), axis=0): v.table.shape[0]
                for k, v in grid.offset_providers.items()
                if hasattr(v, "table")
            },
            **{
                field_size_symbol_name(connectivity_identifier(k), axis=1): v.table.shape[1]
                for k, v in grid.offset_providers.items()
                if hasattr(v, "table")
            },
            **{
                field_stride_symbol_name(
                    connectivity_identifier(k), axis=0
                ): get_stride_from_numpy_to_dace(v.table, 0)
                for k, v in grid.offset_providers.items()
                if hasattr(v, "table")
            },
            **{
                field_stride_symbol_name(
                    connectivity_identifier(k), axis=1
                ): get_stride_from_numpy_to_dace(v.table, 1)
                for k, v in grid.offset_providers.items()
                if hasattr(v, "table")
            },
        }

    def mod_args_for_dace_structures(fuse_func: Callable, args: Any) -> tuple:
        """Modify the args to support DaCe Structures, i.e., teach DaCe how to extract the data from the corresponding GT4Py structures"""
        # DiffusionDiagnosticState_t
        new_args = [
            DiffusionDiagnosticState_t.dtype._typeclass.as_ctypes()(  # noqa: F405
                hdef_ic=k_v[1].hdef_ic.data_ptr(),
                div_ic=k_v[1].div_ic.data_ptr(),
                dwdx=k_v[1].dwdx.data_ptr(),
                dwdy=k_v[1].dwdy.data_ptr(),
            )
            if k_v[0] is DiffusionDiagnosticState_t  # noqa: F405
            else k_v[1]
            for k_v in list(zip(list(fuse_func.__annotations__.values()), list(args), strict=False))
        ]
        # PrognosticState_t
        new_args = [
            PrognosticState_t.dtype._typeclass.as_ctypes()(  # noqa: F405
                rho=k_v[1].rho.data_ptr(),
                w=k_v[1].w.data_ptr(),
                vn=k_v[1].vn.data_ptr(),
                exner=k_v[1].exner.data_ptr(),
                theta_v=k_v[1].theta_v.data_ptr(),
            )
            if k_v[0] is PrognosticState_t  # noqa: F405
            else k_v[1]
            for k_v in list(
                zip(list(fuse_func.__annotations__.values()), list(new_args), strict=False)
            )
        ]
        # OffsetProviders_int64_t
        new_args = [
            OffsetProviders_int64_t.dtype._typeclass.as_ctypes()(  # noqa: F405
                **{
                    member: k_v[1][member].data_ptr()
                    for member in k_v[1].keys()
                    if hasattr(k_v[1][member], "table")
                }
            )
            if k_v[0] is OffsetProviders_int64_t  # noqa: F405
            else k_v[1]
            for k_v in list(
                zip(list(fuse_func.__annotations__.values()), list(new_args), strict=False)
            )
        ]
        # OffsetProviders_int32_t
        new_args = [
            OffsetProviders_int32_t.dtype._typeclass.as_ctypes()(  # noqa: F405
                **{
                    member: k_v[1][member].data_ptr()
                    for member in k_v[1].keys()
                    if hasattr(k_v[1][member], "table")
                }
            )
            if k_v[0] is OffsetProviders_int32_t  # noqa: F405
            else k_v[1]
            for k_v in list(
                zip(list(fuse_func.__annotations__.values()), list(new_args), strict=False)
            )
        ]

        for new_arg_ in new_args:
            if isinstance(new_arg_, DiffusionDiagnosticState_t.dtype._typeclass.as_ctypes()):  # noqa: F405
                new_arg_.descriptor = DiffusionDiagnosticState_t  # noqa: F405
            if isinstance(new_arg_, PrognosticState_t.dtype._typeclass.as_ctypes()):  # noqa: F405
                new_arg_.descriptor = PrognosticState_t  # noqa: F405
            if isinstance(new_arg_, OffsetProviders_int64_t.dtype._typeclass.as_ctypes()):  # noqa: F405
                new_arg_.descriptor = OffsetProviders_int64_t  # noqa: F405
            if isinstance(new_arg_, OffsetProviders_int32_t.dtype._typeclass.as_ctypes()):  # noqa: F405
                new_arg_.descriptor = OffsetProviders_int32_t  # noqa: F405
        return tuple(new_args)

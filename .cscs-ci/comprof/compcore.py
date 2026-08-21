# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile profiler core: JSONL event logging + gt4py monkeypatches.

Active only when ICON4PY_COMPROF=1. Import-safe anywhere: all gt4py patching
happens through a meta-path post-import hook, so importing this module is cheap
and never imports gt4py eagerly.
"""

from __future__ import annotations

import dataclasses
import importlib.abc
import importlib.util
import json
import os
import pathlib
import sys
import threading
import time


_ACTIVE = os.environ.get("ICON4PY_COMPROF") == "1"
_LOG_PATH = os.environ.get("ICON4PY_COMPROF_LOG", "/tmp/comprof.jsonl")
_lock = threading.Lock()


def log_event(**kw):
    if not _ACTIVE:
        return
    kw["pid"] = os.getpid()
    kw["ts"] = round(time.time(), 3)
    line = json.dumps(kw, default=str)
    with _lock, pathlib.Path(_LOG_PATH).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if not _ACTIVE:
    install = lambda: None  # noqa: E731
else:

    class _TimedStep:
        """Wrap one executor sub-step in the worker; created inside the worker only."""

        def __init__(self, inner, phase, prog_name):
            self._inner = inner
            self._phase = phase
            self._prog_name = prog_name

        def __call__(self, inp):
            t0 = time.perf_counter()
            try:
                return self._inner(inp)
            finally:
                log_event(
                    event="step",
                    name=self._prog_name,
                    phase=self._phase,
                    dur=round(time.perf_counter() - t0, 4),
                )

    class TimedExecutor:
        """Picklable (top-level) executor wrapper crossing into the worker.

        Times the whole executor call and, when the executor is an
        OTFCompileWorkflow-like sequence of named steps, each step separately.
        """

        def __init__(self, inner, prog_name):
            self.inner = inner
            self.prog_name = prog_name

        def __call__(self, compilable):
            t0 = time.perf_counter()
            inner = self.inner
            try:
                # named-step sequence: translate / bindings / compilation
                fields = (
                    {f.name for f in dataclasses.fields(inner)}
                    if dataclasses.is_dataclass(inner)
                    else set()
                )
                if {"translation", "bindings", "compilation"} <= fields:
                    inner = dataclasses.replace(
                        inner,
                        translation=_TimedStep(inner.translation, "translation", self.prog_name),
                        bindings=_TimedStep(inner.bindings, "bindings", self.prog_name),
                        compilation=_TimedStep(inner.compilation, "compilation", self.prog_name),
                    )
                return inner(compilable)
            finally:
                log_event(
                    event="executor",
                    name=self.prog_name,
                    dur=round(time.perf_counter() - t0, 4),
                )

    _state = {"patched": set()}

    def _patch_compilation_tasks(mod):
        if "ct" in _state["patched"]:
            return
        _state["patched"].add("ct")
        orig_make = mod.make_compilation_task

        def make_compilation_task(backend, definition_stage, compile_time_args):
            prog = getattr(getattr(definition_stage, "definition", None), "__name__", "?")
            t0 = time.perf_counter()
            task = orig_make(backend, definition_stage, compile_time_args)
            dur = time.perf_counter() - t0  # includes main-side backend.transforms
            log_event(event="make_task", name=prog, dur=round(dur, 4))
            if task.no_offload_reason is None:
                return dataclasses.replace(task, executor=TimedExecutor(task.executor, prog))
            log_event(event="no_offload", name=prog, reason=task.no_offload_reason)
            return task

        mod.make_compilation_task = make_compilation_task

    def _patch_compiled_program(mod):
        if "cp" in _state["patched"]:
            return
        _state["patched"].add("cp")
        cls = mod.CompiledProgramsPool
        orig_cv = cls._compile_variant
        orig_fj = cls._finish_compilation_job

        def _compile_variant(self, **kwargs):
            prog = getattr(getattr(self.definition_stage, "definition", None), "__name__", "?")
            t0 = time.perf_counter()
            ret = orig_cv(self, **kwargs)
            log_event(event="submit", name=prog, dur=round(time.perf_counter() - t0, 4))
            return ret

        def _finish_compilation_job(self, key):
            pending = key in self._compilation_jobs
            t0 = time.perf_counter()
            ret = orig_fj(self, key)
            if pending:
                prog = getattr(getattr(self.definition_stage, "definition", None), "__name__", "?")
                log_event(event="finish", name=prog, dur=round(time.perf_counter() - t0, 4))
            return ret

        cls._compile_variant = _compile_variant
        cls._finish_compilation_job = _finish_compilation_job

    def _patch_auto_optimize(mod):
        if "ao" in _state["patched"]:
            return
        _state["patched"].add("ao")
        orig = mod.gt_auto_optimize

        def gt_auto_optimize(sdfg, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(sdfg, *args, **kwargs)
            finally:
                log_event(
                    event="autoopt",
                    name=getattr(sdfg, "name", "?"),
                    dur=round(time.perf_counter() - t0, 4),
                )

        mod.gt_auto_optimize = gt_auto_optimize

    def _patch_locking(mod):
        if "lk" in _state["patched"]:
            return
        _state["patched"].add("lk")
        orig_lock = mod.lock

        class _TimedLock:
            def __init__(self, cm, target):
                self._cm = cm
                self._target = str(target)

            def __enter__(self):
                t0 = time.perf_counter()
                r = self._cm.__enter__()
                dur = time.perf_counter() - t0
                if dur > 0.05:
                    log_event(event="lock_wait", name=self._target, dur=round(dur, 4))
                return r

            def __exit__(self, *a):
                return self._cm.__exit__(*a)

        def lock(target):
            return _TimedLock(orig_lock(target), target)

        mod.lock = lock

    def _patch_dace(mod):
        if "dace" in _state["patched"]:
            return
        _state["patched"].add("dace")
        SDFG = mod.SDFG

        orig_from_json = SDFG.from_json.__func__

        @classmethod
        def from_json(cls, json_obj, *a, **k):
            t0 = time.perf_counter()
            try:
                return orig_from_json(cls, json_obj, *a, **k)
            finally:
                log_event(event="dace_from_json", dur=round(time.perf_counter() - t0, 4))

        SDFG.from_json = from_json

        orig_compile = SDFG.compile

        def timed_sdfg_compile(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return orig_compile(self, *a, **k)
            finally:
                log_event(
                    event="dace_sdfg_compile",
                    name=getattr(self, "name", "?"),
                    dur=round(time.perf_counter() - t0, 4),
                )

        SDFG.compile = timed_sdfg_compile

    def _patch_dace_codegen(mod):
        if "dcg" in _state["patched"]:
            return
        _state["patched"].add("dcg")
        for fn_name, ev in (
            ("generate_program_folder", "dace_prog_folder"),
            ("configure_and_compile", "dace_configure_build"),
        ):
            orig = getattr(mod, fn_name, None)
            if orig is None:
                continue

            def make_wrapped(orig, ev):
                def wrapped(*args, **kwargs):
                    t0 = time.perf_counter()
                    try:
                        return orig(*args, **kwargs)
                    finally:
                        log_event(event=ev, dur=round(time.perf_counter() - t0, 4))

                return wrapped

            setattr(mod, fn_name, make_wrapped(orig, ev))

    def _patch_dace_codegen_mod(mod):
        if "dgc" in _state["patched"]:
            return
        _state["patched"].add("dgc")
        orig = getattr(mod, "generate_code", None)
        if orig is not None:

            def generate_code(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return orig(*args, **kwargs)
                finally:
                    log_event(event="dace_codegen", dur=round(time.perf_counter() - t0, 4))

            mod.generate_code = generate_code

    _PATCHES = {
        "gt4py.next.otf.compilation_tasks": _patch_compilation_tasks,
        "gt4py.next.otf.compiled_program": _patch_compiled_program,
        "gt4py.next.program_processors.runners.dace.transformations.auto_optimize": _patch_auto_optimize,
        "gt4py._core.locking": _patch_locking,
        "dace": _patch_dace,
        "dace.codegen.compiler": _patch_dace_codegen,
        "dace.codegen.codegen": _patch_dace_codegen_mod,
    }

    class _CompProfFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in _PATCHES:
                return importlib.machinery.ModuleSpec(fullname, loader=_PostImportLoader(fullname))
            return None

    class _PostImportLoader(importlib.abc.Loader):
        def __init__(self, fullname):
            self.fullname = fullname

        def create_module(self, spec):
            return None  # default creation

        def exec_module(self, module):
            _PATCHES[self.fullname](module)

    # NOTE: returning a ModuleSpec with a *different* loader from find_spec would
    # steal loading from the normal finder. Instead, we hook module exec via
    # a simpler approach: wrap at import completion using a loader is wrong;
    # use the "audit hook" free approach below.
    def install():
        # Simpler and robust: patch on first attribute access is fragile for
        # module objects; use a tiny import hook that delegates spec creation to
        # the remaining finders and only wraps exec_module.

        class _DelegatingFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname not in _PATCHES:
                    return None
                # find the real spec without our hook in the way
                sys.meta_path.remove(self)
                try:
                    spec = importlib.util.find_spec(fullname)
                finally:
                    sys.meta_path.insert(0, self)
                if spec is None or spec.loader is None:
                    return None
                orig_loader = spec.loader
                patch = _PATCHES[fullname]

                class _WrappingLoader(importlib.abc.Loader):
                    def create_module(self, spec2):
                        if hasattr(orig_loader, "create_module"):
                            return orig_loader.create_module(spec2)
                        return None

                    def exec_module(self, module):
                        orig_loader.exec_module(module)
                        patch(module)

                spec.loader = _WrappingLoader()
                return spec

        sys.meta_path.insert(0, _DelegatingFinder())
        # if already imported (e.g. gt4py imported before sitecustomize somehow),
        # patch immediately.
        for name, patch in _PATCHES.items():
            mod = sys.modules.get(name)
            if mod is not None:
                patch(mod)

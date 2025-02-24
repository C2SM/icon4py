import pytest
import os

GT4PY_BACKEND = os.environ.get("GT4PY_BACKEND", None)
if GT4PY_BACKEND is None:
    print("Error: GT4PY_BACKEND environment variable is not set.")
    exit(1)
ICON4PY_GRID = os.environ.get("ICON4PY_GRID", None)
if ICON4PY_GRID is None:
    print("Error: ICON4PY_GRID environment variable is not set.")
    exit(1)
BENCHMARK_DIR = os.environ.get("ASV_BUILD_DIR", None)
if BENCHMARK_DIR is None:
    print("Error: ASV_BUILD_DIR environment variable is not set.")
    exit(1)
COMMIT_HASH = os.environ.get("ASV_COMMIT", None)
if COMMIT_HASH is None:
    print("Error: COMMIT_HASH environment variable is not set.")
    exit(1)
MEMRAY = os.environ.get("MEMRAY", None)
BENCHMARK_RUNTIME_FILENAME = "benchmark_runtime_{}_{}_{}.json".format(COMMIT_HASH, GT4PY_BACKEND, ICON4PY_GRID)
BENCHMARK_MEMRAY_FILENAME = "benchmark_memray_{}_{}_{}.json".format(COMMIT_HASH, GT4PY_BACKEND, ICON4PY_GRID)

benchmark_runtime_file_path = os.path.join(BENCHMARK_DIR, BENCHMARK_RUNTIME_FILENAME) if BENCHMARK_DIR else BENCHMARK_RUNTIME_FILENAME
benchmark_memray_file_path = os.path.join(BENCHMARK_DIR, BENCHMARK_MEMRAY_FILENAME) if BENCHMARK_DIR else BENCHMARK_MEMRAY_FILENAME

if os.environ.get("MEMRAY", None) is None:
    pytest.main([os.path.join(os.path.dirname(__file__), "../model/atmosphere/dycore/tests"), "--benchmark-json", benchmark_runtime_file_path, "--benchmark-only", "--backend", GT4PY_BACKEND, "--grid", ICON4PY_GRID, "-k", "test_fused_velocity_advection_stencil_15_to_18", "--benchmark-min-rounds=1"])
else:
    pytest.main([os.path.join(os.path.dirname(__file__), "../model/atmosphere/dycore/tests"), "--benchmark-json", benchmark_memray_file_path, "--benchmark-only", "--backend", GT4PY_BACKEND, "--grid", ICON4PY_GRID, "-k", "test_fused_velocity_advection_stencil_15_to_18", "--benchmark-min-rounds=1", "--memray"])


import pytest
import json
import os
import traceback

BENCHMARKS = {}
PREFIX = "track_"
BENCHMARK_DIR = os.environ.get("ASV_BUILD_DIR", os.path.join(os.environ.get("ASV_ENV_DIR"), "project"))
COMMIT_HASH = os.environ.get("ASV_COMMIT", None)
GT4PY_BACKEND = os.environ.get("GT4PY_BACKEND", None)
ICON4PY_GRID = os.environ.get("ICON4PY_GRID", None)
BENCHMARK_RUNTIME_FILENAME = "benchmark_runtime_{}_{}_{}.json".format(COMMIT_HASH, GT4PY_BACKEND, ICON4PY_GRID)
BENCHMARK_MEMRAY_FILENAME = "benchmark_memray_{}_{}_{}.json".format(COMMIT_HASH, GT4PY_BACKEND, ICON4PY_GRID)
MEMRAY = os.environ.get("MEMRAY", None)

benchmark_runtime_file_path = os.path.join(BENCHMARK_DIR, BENCHMARK_RUNTIME_FILENAME) if BENCHMARK_DIR else BENCHMARK_RUNTIME_FILENAME
benchmark_memray_file_path = os.path.join(BENCHMARK_DIR, BENCHMARK_MEMRAY_FILENAME) if BENCHMARK_DIR else BENCHMARK_MEMRAY_FILENAME

if MEMRAY is not None:
    if not os.path.exists(benchmark_memray_file_path):
        print(f"Error: Benchmark memray file {benchmark_memray_file_path} does not exist.")
        exit(1)
    else:
        with open(benchmark_memray_file_path, "r") as f:
            benchmark_data = json.load(f)

            for benchmark in benchmark_data["benchmarks"]:
                benchmark_name = benchmark["name"]
                if GT4PY_BACKEND in benchmark_name:
                    if ICON4PY_GRID in benchmark_name:
                        filtered_name = benchmark_name.replace("-", "_").replace("=", "_").replace("[", "_").replace("]", "_").replace(f"_benchmark_backend_{GT4PY_BACKEND}_", "_").replace(f"_grid_{ICON4PY_GRID}_", "_").replace("test_", "_")
                        print("Filtered name: {}".format(filtered_name))
                        mem_high_watermark = benchmark["extra_info"]["memory_high_watermark"]
                        asv_mem_name = "{}memory{}".format(PREFIX, filtered_name)
                        print("Asv mem name: {}".format(asv_mem_name))
                        def asv_memray_method(self, mem_high_watermark=mem_high_watermark): return mem_high_watermark
                        asv_memray_method.unit = "MB"
                        asv_memray_method.number = 1
                        asv_memray_method.repeat = 1
                        BENCHMARKS[asv_mem_name] = asv_memray_method
        benchmark_runtime_file_path = benchmark_memray_file_path

if not os.path.exists(benchmark_runtime_file_path):
    print(f"Error: Benchmark runtime file {benchmark_runtime_file_path} does not exist.")
    exit(1)

with open(benchmark_runtime_file_path, "r") as f:
    benchmark_data = json.load(f)

    for benchmark in benchmark_data["benchmarks"]:
        benchmark_name = benchmark["name"]
        if GT4PY_BACKEND in benchmark_name:
            if ICON4PY_GRID in benchmark_name:
                filtered_name = benchmark_name.replace("-", "_").replace("=", "_").replace("[", "_").replace("]", "_").replace(f"_benchmark_backend_{GT4PY_BACKEND}_", "_").replace(f"_grid_{ICON4PY_GRID}_", "_").replace("test_", "_")
                print("Filtered name: {}".format(filtered_name))
                time = benchmark["stats"]["median"]
                asv_runtime_name = "{}runtime{}".format(PREFIX, filtered_name)
                print("Asv runtime name: {}".format(asv_runtime_name))
                def asv_runtime_method(self, t=time): return t if MEMRAY is None else None
                asv_runtime_method.unit = "s"
                asv_runtime_method.number = 1
                asv_runtime_method.repeat = 1
                BENCHMARKS[asv_runtime_name] = asv_runtime_method
                if MEMRAY is None:
                    asv_mem_name = "{}memory{}".format(PREFIX, filtered_name)
                    def asv_memray_method(self): return None
                    asv_memray_method.unit = "MB"
                    asv_memray_method.number = 1
                    asv_memray_method.repeat = 1
                    BENCHMARKS[asv_mem_name] = asv_memray_method

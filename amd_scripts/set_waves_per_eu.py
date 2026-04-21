"""
DaCe post-compilation patch: add __attribute__((amdgpu_waves_per_eu(min, max)))
to all __global__ kernels in a GT4Py DaCe cache directory.

This is a source-level patch applied after DaCe generates HIP code but before
(or instead of) the final compilation step. It works by sed-replacing kernel
declarations in the generated .cpp file.

Usage:
    python amd_scripts/set_waves_per_eu.py <cache_dir> [--min-waves 1] [--max-waves 4]

Example:
    python amd_scripts/set_waves_per_eu.py amd_profiling_solver_regional --min-waves 1 --max-waves 4

After patching, delete the build/ directory to force recompilation:
    rm -rf <cache_dir>/.gt4py_cache/*/build/
Then re-run the benchmark.
"""

import argparse
import glob
import re
import shutil
import sys
from pathlib import Path


def patch_kernels(cpp_file: Path, min_waves: int, max_waves: int, dry_run: bool = False) -> int:
    """Add amdgpu_waves_per_eu attribute to all __global__ kernel declarations."""
    content = cpp_file.read_text()

    attr = f"__attribute__((amdgpu_waves_per_eu({min_waves}, {max_waves})))"

    # Match: __global__ void [__launch_bounds__(...)] kernel_name(
    # Replace with: __global__ void [__launch_bounds__(...)] __attribute__(...) kernel_name(
    pattern = r"(__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?)(\w+\s*\()"

    def replacement(m):
        prefix = m.group(1)
        kernel_start = m.group(2)
        # Don't add if already patched
        if "amdgpu_waves_per_eu" in prefix:
            return m.group(0)
        return f"{prefix}{attr} {kernel_start}"

    new_content, count = re.subn(pattern, replacement, content)

    if count > 0 and not dry_run:
        backup = cpp_file.with_suffix(".cpp.orig")
        if not backup.exists():
            shutil.copy2(cpp_file, backup)
        cpp_file.write_text(new_content)

    return count


def main():
    parser = argparse.ArgumentParser(description="Patch DaCe-generated HIP kernels with amdgpu_waves_per_eu")
    parser.add_argument("cache_dir", help="GT4Py build cache directory")
    parser.add_argument("--min-waves", type=int, default=1, help="Minimum waves per EU (default: 1)")
    parser.add_argument("--max-waves", type=int, default=4, help="Maximum waves per EU (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Error: {cache_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all generated HIP/CUDA cpp files
    cpp_files = list(cache_dir.glob(".gt4py_cache/*/src/cuda/**/*.cpp"))
    if not cpp_files:
        # Try without hip subdirectory
        cpp_files = list(cache_dir.glob(".gt4py_cache/*/src/cuda/*.cpp"))

    if not cpp_files:
        print(f"No generated .cpp files found in {cache_dir}", file=sys.stderr)
        sys.exit(1)

    total_patched = 0
    for cpp_file in cpp_files:
        count = patch_kernels(cpp_file, args.min_waves, args.max_waves, args.dry_run)
        if count > 0:
            action = "Would patch" if args.dry_run else "Patched"
            print(f"{action} {count} kernels in {cpp_file}")
            total_patched += count

    if total_patched == 0:
        print("No kernels found to patch (already patched or no __global__ functions)")
    else:
        action = "Would patch" if args.dry_run else "Patched"
        print(f"\n{action} {total_patched} kernels total.")
        if not args.dry_run:
            print(f"\nNow delete build dirs to force recompilation:")
            print(f"  rm -rf {cache_dir}/.gt4py_cache/*/build/")
            print(f"Then re-run the benchmark.")


if __name__ == "__main__":
    main()

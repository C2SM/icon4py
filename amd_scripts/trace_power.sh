#!/usr/bin/env bash
# Sample GPU power/clocks at 100 ms while a benchmark runs, so we can compare
# under-load behavior across clusters (beverin vs aac6) instead of idle state.
#
# Usage:
#   bash amd_scripts/trace_power.sh <out-prefix> -- <command to run>
#
# Example:
#   srun --partition=mi300 --gres=gpu:1 --exclusive --ntasks=1 --time=02:00:00 \
#       bash amd_scripts/trace_power.sh beverin_run1 -- \
#       .venv_rocm/bin/python -m pytest -sv -m continuous_benchmarking -p no:tach \
#           --backend=dace_gpu --grid=icon_benchmark_regional \
#           model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
#           -k "test_TestVerticallyImplicitSolverAtPredictorStep[..."
#
# Outputs (in cwd):
#   <prefix>_<host>_power.csv     # one row per sample, comma-separated
#   <prefix>_<host>_bench.log     # stdout/stderr of the benchmark command
#   <prefix>_<host>_summary.txt   # min/median/p95/max for power & mclk during the busy window
#
# Why this script (and not just rocm-smi --csv in a loop):
#   - Forces consistent column set across rocm-smi versions
#   - Tags each sample with epoch ms so we can window the busy phase later
#   - Computes summary stats so you don't have to eyeball thousands of rows

set -u

if [ $# -lt 2 ] || [ "$2" != "--" ] && [ "$1" != "-h" ]; then
    cat <<EOF >&2
Usage: $0 <out-prefix> -- <command...>
       $0 -h
EOF
    exit 2
fi

if [ "$1" = "-h" ]; then
    sed -n '2,25p' "$0"
    exit 0
fi

PREFIX="$1"
shift
shift  # drop the literal "--"

HOST="$(hostname -s)"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="${PREFIX}_${HOST}_${TS}_power.csv"
LOG="${PREFIX}_${HOST}_${TS}_bench.log"
SUM="${PREFIX}_${HOST}_${TS}_summary.txt"

# Pick a sampler. Prefer amd-smi (newer, stable JSON) but fall back to rocm-smi.
SAMPLER=""
if command -v amd-smi >/dev/null 2>&1; then
    SAMPLER="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    SAMPLER="rocm-smi"
else
    echo "ERROR: neither amd-smi nor rocm-smi on PATH" >&2
    exit 3
fi

# Which GPU is this job allocated? Default to 0 if unset.
GPU_IDX="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}}"
GPU_IDX="${GPU_IDX%%,*}"

echo "trace_power: sampler=$SAMPLER gpu=$GPU_IDX host=$HOST"
echo "trace_power: csv=$CSV log=$LOG"

# Header
echo "epoch_ms,gpu,sclk_mhz,mclk_mhz,fclk_mhz,power_w,temp_c,gfx_busy_pct,mem_busy_pct,perf_level" > "$CSV"

# Sampler loop in background. Using a subshell so we can kill cleanly.
(
    while :; do
        NOW_MS=$(date +%s%3N)
        if [ "$SAMPLER" = "amd-smi" ]; then
            # amd-smi metric -g <id> ... gives JSON-ish; we parse the lines we need.
            OUT=$(amd-smi metric -g "$GPU_IDX" --clock --power --usage --temperature 2>/dev/null)
            SCLK=$(echo "$OUT" | awk '/GFX_0/{f=1} f && /CLK:/{print $2; exit}')
            MCLK=$(echo "$OUT" | awk '/MEM_0/{f=1} f && /CLK:/{print $2; exit}')
            FCLK=$(echo "$OUT" | awk '/FABRIC/{f=1} f && /CLK:/{print $2; exit}')
            POWER=$(echo "$OUT" | awk '/SOCKET_POWER/{print $2; exit}')
            TEMP=$(echo "$OUT" | awk '/HOTSPOT_TEMPERATURE/{print $2; exit}')
            GFX=$(echo "$OUT" | awk '/GFX_ACTIVITY/{print $2; exit}')
            MEM=$(echo "$OUT" | awk '/UMC_ACTIVITY/{print $2; exit}')
            PERF="auto"
        else
            # rocm-smi path
            OUT=$(rocm-smi -d "$GPU_IDX" -P -c -g -t -u --showperflevel 2>/dev/null)
            SCLK=$(echo "$OUT" | awk -F'[():]' '/sclk clock/{gsub(/Mhz/,"",$0); print $4; exit}')
            MCLK=$(echo "$OUT" | awk -F'[():]' '/mclk clock/{gsub(/Mhz/,"",$0); print $4; exit}')
            FCLK=$(echo "$OUT" | awk -F'[():]' '/fclk clock/{gsub(/Mhz/,"",$0); print $4; exit}')
            POWER=$(echo "$OUT" | awk -F: '/Average Graphics Package Power|Current Socket Graphics Package Power/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
            TEMP=$(echo "$OUT" | awk -F: '/Temperature \(Sensor edge\)|Temperature \(Sensor junction\)/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
            GFX=$(echo "$OUT" | awk -F: '/GPU use \(%\)/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
            MEM=$(echo "$OUT" | awk -F: '/GPU Memory use/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
            PERF=$(echo "$OUT" | awk -F: '/Performance Level/{gsub(/^ +| +$/,"",$2); print $2; exit}')
        fi
        echo "${NOW_MS},${GPU_IDX},${SCLK:-},${MCLK:-},${FCLK:-},${POWER:-},${TEMP:-},${GFX:-},${MEM:-},${PERF:-}" >> "$CSV"
        sleep 0.1
    done
) &
SAMPLER_PID=$!

# Make sure we always stop the sampler.
trap 'kill $SAMPLER_PID 2>/dev/null; wait $SAMPLER_PID 2>/dev/null' EXIT INT TERM

# Run the actual benchmark, tee to log so user sees progress.
echo "trace_power: running: $*"
START_MS=$(date +%s%3N)
"$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
END_MS=$(date +%s%3N)

# Stop sampler.
kill $SAMPLER_PID 2>/dev/null
wait $SAMPLER_PID 2>/dev/null
trap - EXIT INT TERM

# Summary: filter to busy window (gfx_busy >= 50%) to skip Python warmup time.
{
    echo "host: $HOST"
    echo "sampler: $SAMPLER"
    echo "command: $*"
    echo "exit_code: $RC"
    echo "wallclock_ms: $((END_MS - START_MS))"
    echo "samples_total: $(($(wc -l < "$CSV") - 1))"
    echo
    echo "=== under-load stats (rows with gfx_busy_pct >= 50) ==="
    awk -F, 'NR>1 && $8+0 >= 50 {print $0}' "$CSV" > /tmp/_busy.$$
    BUSY=$(wc -l < /tmp/_busy.$$)
    echo "samples_busy: $BUSY"
    if [ "$BUSY" -gt 0 ]; then
        # Compute min/p50/p95/max in awk in one pass per column.
        # Skips empty / non-numeric values (some metrics may be unavailable).
        for col_name_pair in "sclk_mhz:3" "mclk_mhz:4" "fclk_mhz:5" "power_w:6" "temp_c:7" "gfx_busy_pct:8" "mem_busy_pct:9"; do
            name="${col_name_pair%%:*}"
            col="${col_name_pair##*:}"
            awk -F, -v name="$name" -v col="$col" '
                $col ~ /^[+-]?[0-9]+(\.[0-9]+)?$/ { a[++n] = $col + 0 }
                END {
                    if (n == 0) {
                        printf "  %-15s (no numeric samples)\n", name
                        exit
                    }
                    # Sort numerically (insertion sort — fine for n < 1000)
                    for (i = 2; i <= n; i++) {
                        v = a[i]; j = i
                        while (j > 1 && a[j-1] > v) { a[j] = a[j-1]; j-- }
                        a[j] = v
                    }
                    p50_idx = int(n * 0.5);  if (p50_idx < 1) p50_idx = 1
                    p95_idx = int(n * 0.95); if (p95_idx < 1) p95_idx = 1
                    printf "  %-15s min=%-8s p50=%-8s p95=%-8s max=%-8s\n", \
                        name, a[1], a[p50_idx], a[p95_idx], a[n]
                }' /tmp/_busy.$$
        done
    else
        echo "  (no busy samples — kernel may be too short for 100 ms cadence;"
        echo "   inspect ${CSV} manually around the kernel-launch window)"
    fi
    rm -f /tmp/_busy.$$
} > "$SUM"

echo "trace_power: done"
echo "  csv:     $CSV"
echo "  log:     $LOG"
echo "  summary: $SUM"
echo
cat "$SUM"

exit $RC

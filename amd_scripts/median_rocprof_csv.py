#!/usr/bin/env python3
import csv
import sys
import statistics

if len(sys.argv) < 2:
    print("Usage: python script.py <csv_file>", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
kernels = {}

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["Kernel_Name"]
        if name.startswith("map"):
            name = name.split("(")[0]
            if name not in kernels:
                kernels[name] = []
            duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            kernels[name].append(duration)

if not kernels:
    print("No kernels starting with 'map' found", file=sys.stderr)
else:
    for kernel_name, durations in sorted(kernels.items()):
        median = statistics.median(durations)
        stdev = statistics.stdev(durations) if len(durations) > 1 else 0
        print(f"{kernel_name},{median:.0f},{stdev:.0f}")

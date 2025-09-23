#!/usr/bin/env python3
import subprocess, time, csv, os
from datetime import datetime

COMMANDS = [
    "python benchmarks/hpatches/hpatches_benchmark.py",
    "python benchmarks/speed_and_memory/speed.py",
    "python benchmarks/benchmark_parallel.py --ds md",
    "python benchmarks/benchmark_parallel.py --ds ghr --ghr-partial",
    "python benchmarks/benchmark_parallel.py --ds sc",
    "python benchmarks/imc/imc_benchmark.py --scene-set val",
]

OUT_DIR = "timing_logs"
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUT_DIR, f"timings_{stamp}.csv")

results = []
print(f"[timing] writing CSV to {csv_path}")

for cmd in COMMANDS:
    print(f"\n=== Running: {cmd}")
    start = time.perf_counter()
    # No stdout/stderr capture: they stream directly to your console
    proc = subprocess.run(cmd, shell=True)
    elapsed = time.perf_counter() - start
    print(f"--- Elapsed: {elapsed:.3f}s (exit {proc.returncode})")
    results.append(
        {"command": cmd, "seconds": f"{elapsed:.3f}", "exit_code": proc.returncode}
    )

# Save minimal CSV: command,seconds,exit_code
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["command", "seconds", "exit_code"])
    w.writeheader()
    w.writerows(results)

print("\nSummary (seconds):")
for r in results:
    print(f"{r['seconds']:>8}  exit={r['exit_code']}  {r['command']}")
print(f"\nCSV saved: {csv_path}")

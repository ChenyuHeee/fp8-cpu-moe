#!/usr/bin/env python3
"""Single-socket DRAM-throughput benchmark for the block-FP8 GEMV."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import statistics
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build" / "libfp8_moe.so"


def _parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            first, last = (int(x) for x in part.split("-", 1))
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(part))
    return cpus


def _physical_cpus_for_node(node: int) -> list[int]:
    node_file = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not node_file.exists():
        raise RuntimeError(f"NUMA node {node} does not exist")
    allowed = set(os.sched_getaffinity(0))
    node_cpus = [cpu for cpu in _parse_cpu_list(node_file.read_text())
                 if cpu in allowed]
    physical: dict[tuple[int, int], int] = {}
    for cpu in node_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        physical.setdefault((package, core), cpu)
    return list(physical.values())


def _cache_size_bytes(cpu: int, level: int = 3) -> int:
    cache_root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    for index in cache_root.glob("index*"):
        if int((index / "level").read_text()) != level:
            continue
        size = (index / "size").read_text().strip().upper()
        multiplier = 1024 if size.endswith("K") else 1024 * 1024
        if size[-1] in "KM":
            size = size[:-1]
        return int(size) * multiplier
    return 0


def _load_library():
    if not LIBRARY.exists():
        raise FileNotFoundError(f"{LIBRARY} does not exist; run `make` first")
    lib = ctypes.CDLL(str(LIBRARY))
    lib.fp8_moe_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.fp8_moe_gemv.restype = ctypes.c_int
    lib.fp8_moe_isa_name.restype = ctypes.c_char_p
    return lib


def _run(lib, weights: np.ndarray, scales: np.ndarray, x: np.ndarray,
         output: np.ndarray, threads: int) -> None:
    rc = lib.fp8_moe_gemv(
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.shape[0],
        weights.shape[1],
        threads,
    )
    if rc != 0:
        raise RuntimeError(f"fp8_moe_gemv returned {rc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, default=0, help="NUMA node/socket to use")
    parser.add_argument("--threads", type=int, default=0,
                        help="physical workers (default: every core on the node)")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--working-set-mib", type=int, default=1024,
                        help="minimum weight matrix size")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=11)
    args = parser.parse_args()
    if args.k <= 0 or args.working_set_mib <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("K, working-set size, and iterations must be positive")

    physical_cpus = _physical_cpus_for_node(args.node)
    if not physical_cpus:
        raise RuntimeError(f"no allowed physical CPUs found on NUMA node {args.node}")
    threads = args.threads or len(physical_cpus)
    if threads < 1 or threads > len(physical_cpus):
        parser.error(f"--threads must be in [1, {len(physical_cpus)}] for node {args.node}")
    selected_cpus = physical_cpus[:threads]

    # The C++ pool reads this affinity mask and pins worker i to CPU i. Allocation
    # happens afterward, so the initialization below first-touches all pages on
    # this node as well.
    os.sched_setaffinity(0, set(selected_cpus))
    target_bytes = args.working_set_mib * 1024 * 1024
    rows = max(128, (target_bytes + args.k - 1) // args.k)
    rows = (rows + 127) // 128 * 128
    k_blocks = (args.k + 127) // 128

    weights = np.empty((rows, args.k), dtype=np.uint8)
    weights.fill(0x38)  # finite e4m3 +1; values do not change the instruction path
    scales = np.full((rows // 128, k_blocks), 127, dtype=np.uint8)
    x = np.linspace(0.25, 1.0, args.k, dtype=np.float32)
    output = np.full(rows, np.nan, dtype=np.float32)
    bytes_per_pass = weights.nbytes + scales.nbytes
    llc_bytes = _cache_size_bytes(selected_cpus[0])
    if llc_bytes and bytes_per_pass <= llc_bytes:
        raise RuntimeError(
            f"working set {bytes_per_pass / 2**20:.1f} MiB does not exceed "
            f"the detected {llc_bytes / 2**20:.1f} MiB LLC"
        )

    lib = _load_library()
    for _ in range(args.warmup):
        output.fill(np.nan)
        _run(lib, weights, scales, x, output, threads)

    durations = []
    for _ in range(args.iterations):
        output.fill(np.nan)
        start = time.perf_counter_ns()
        _run(lib, weights, scales, x, output, threads)
        durations.append((time.perf_counter_ns() - start) * 1e-9)
    if not np.isfinite(output).all() or float(output.sum()) == 0.0:
        raise RuntimeError("invalid output checksum")

    bandwidths = [bytes_per_pass / elapsed / 1e9 for elapsed in durations]
    median = statistics.median(bandwidths)
    median_ms = statistics.median(durations) * 1e3
    criterion = median >= 50.6
    print(f"ISA:             {lib.fp8_moe_isa_name().decode('ascii')}")
    print(f"NUMA node:       {args.node}")
    print(f"physical cores:  {threads} ({selected_cpus[0]}..{selected_cpus[-1]})")
    print(f"shape:           N={rows}, K={args.k}")
    print(f"working set:     {bytes_per_pass / 2**20:.2f} MiB")
    print(f"detected LLC:    {llc_bytes / 2**20:.2f} MiB")
    print(f"samples:         {args.iterations} after {args.warmup} warmups")
    print(f"median latency:  {median_ms:.3f} ms")
    print(f"median:          {median:.2f} GB/s")
    print(f"range:           {min(bandwidths):.2f} .. {max(bandwidths):.2f} GB/s")
    print(f"criterion:       {'PASS' if criterion else 'FAIL'} (>= 50.6 GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

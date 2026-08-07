#!/usr/bin/env python3
"""measure_gate_workers.py — what the parallel half costs at N workers.

F219 follow-up. `-n auto` was chosen on a machine running the **cpu** install profile.
The gate the orchestrator runs before a merge uses the **gpu** one, where `import torch`
is half a gigabyte instead of a few dozen megabytes — and 24 workers of that ran the
machine out of memory (`MemoryError` in `hashing.py`, on a 1 MiB read).

So the worker count is not a question about cores, it is a question about memory per
process, and the answer has to be measured on the profile that pays the most for it.
This script runs the parallel half at each N and reports two numbers per run: wall clock
and the peak resident memory of the run's OWN process tree. Its own tree, not every
python on the box, because the box is shared — other worktrees run their gates on it and
the product itself gets run on it, and a measurement that counts their memory as ours is
a measurement of who else was awake.

    uv run --extra gpu --extra dev python scripts/measure_gate_workers.py 4 8 12 24

Not part of the gate: it takes a few minutes per N. It lives here for the same reason
the other measure_* modules do — the number in `check.py` has to be re-derivable.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_char * 260),
    ]


def _parents() -> dict[int, int]:
    """pid -> parent pid, for every process on the machine."""
    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    entry = _ProcessEntry()
    entry.dwSize = ctypes.sizeof(_ProcessEntry)
    tree: dict[int, int] = {}
    try:
        more = kernel32.Process32First(snapshot, ctypes.byref(entry))
        while more:
            tree[entry.th32ProcessID] = entry.th32ParentProcessID
            more = kernel32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return tree


def _resident_mb(pid: int) -> float:
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    handle = kernel32.OpenProcess(0x1000, False, pid)  # LIMITED_INFORMATION
    if not handle:
        return 0.0
    try:
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0.0
        return counters.WorkingSetSize / 2 ** 20
    finally:
        kernel32.CloseHandle(handle)


def tree_memory_mb(root: int) -> tuple[float, int]:
    """(resident MB of `root` and everything under it, how many processes that is).

    Read through the process table rather than through a library, so this needs nothing
    that is not already installed to run the gate.
    """
    if os.name != "nt":  # pragma: no cover - the measurement is a Windows one
        out = subprocess.run(["ps", "-eo", "pid,ppid,rss"], capture_output=True,
                             text=True).stdout
        rows = [line.split() for line in out.splitlines()[1:] if len(line.split()) == 3]
        parents = {int(pid): int(ppid) for pid, ppid, _rss in rows}
        resident = {int(pid): int(rss) / 1024 for pid, _ppid, rss in rows}
    else:
        parents = _parents()
        resident = None  # read lazily below: opening every process on the box is waste

    def descends_from(pid: int) -> bool:
        seen = set()
        while pid and pid not in seen:
            if pid == root:
                return True
            seen.add(pid)
            pid = parents.get(pid, 0)
        return False

    ours = [pid for pid in parents if descends_from(pid)]
    if resident is not None:  # pragma: no cover - the measurement is a Windows one
        return sum(resident.get(pid, 0.0) for pid in ours), len(ours)
    return sum(_resident_mb(pid) for pid in ours), len(ours)


def run_at(workers: int, extra: list[str], log_dir: str) -> tuple[float, float, int]:
    """(seconds, peak MB of the run's process tree, exit code) for one parallel half.

    The output is kept: a non-zero exit here is the interesting result, and "it failed
    at 12 workers" without the reason is not a measurement of anything.
    """
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-cov", "-m", "not serial",
           "-n", str(workers), "--dist", "loadfile", *extra]
    started = time.monotonic()
    log = open(os.path.join(log_dir, f"workers-{workers:02d}.log"), "wb")
    proc = subprocess.Popen(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT)
    peak = 0.0
    while proc.poll() is None:
        resident, _count = tree_memory_mb(proc.pid)
        peak = max(peak, resident)
        time.sleep(2)
    log.close()
    return time.monotonic() - started, peak, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("workers", nargs="*", type=int, default=[4, 8, 12, 16, 24],
                        help="worker counts to try (default: 4 8 12 16 24)")
    parser.add_argument("--pytest-arg", action="append", default=[],
                        help="extra argument for pytest, repeatable")
    parser.add_argument("--logs", default=os.path.join(_ROOT, ".measure"),
                        help="where to keep each run's pytest output")
    args = parser.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    print(f"{'workers':>8} {'seconds':>9} {'peak MB':>10} {'MB/worker':>10} {'exit':>5}",
          flush=True)
    for workers in args.workers:
        seconds, peak, code = run_at(workers, args.pytest_arg, args.logs)
        # Flushed per row: the sweep takes tens of minutes, and a table that appears
        # only at the end cannot be watched.
        print(f"{workers:>8} {seconds:>9.1f} {peak:>10.0f} {peak / workers:>10.0f} {code:>5}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

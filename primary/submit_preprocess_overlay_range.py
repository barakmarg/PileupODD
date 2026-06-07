"""
Run create_training_dataset_pileup_overlay.run_preprocessing_pipeline over
a contiguous HS file range, splitting the range into 3-file groups that
match the PU 3-file blocks. Each group runs in its OWN subprocess (so
polars + GPU + temp-dir memory is fully released between groups) and the
subprocess output is streamed live to the manager's stdout.

PU block mapping (per HS file `i`):
    block_base = (i // 3) * 3
    pu_indices for that block = [block_base, block_base+1, block_base+2]
So HS [0,1,2] → PU [0,1,2]; HS [3,4,5] → PU [3,4,5]; HS [0] → PU [0,1,2].

Group splitting:
    HS [0, 100) with --group-size 3 → groups
      [0,1,2], [3,4,5], …, [96,97,98], [99]
    each group spawned sequentially.

Usage (manager):
    python submit_preprocess_overlay_range.py --hs-start 0 --hs-end 100
    python submit_preprocess_overlay_range.py --hs-start 0 --hs-end 6 --chunk-size 50

Worker mode (--worker) is invoked internally by the manager — don't call
directly unless debugging.
"""

import argparse
import subprocess
import sys
import time


def _pu_pool_for(hs_indices: list[int]) -> list[int]:
    """Union of 3-file PU blocks covering the given HS file indices."""
    blocks = sorted({(i // 3) * 3 for i in hs_indices})
    return sorted({b + k for b in blocks for k in (0, 1, 2)})


def _run_worker(hs_start: int, hs_end: int, event_name: str,
                chunk_size: int) -> None:
    """Subprocess entry: load run_preprocessing_pipeline (heavy import) and
    process exactly the HS files in [hs_start, hs_end)."""
    sys.path.insert(0, '/storage/agrp/barakma/PileupODD')
    from primary.create_training_dataset_pileup_overlay import (  # noqa: E402
        run_preprocessing_pipeline,
    )

    hs_indices = list(range(hs_start, hs_end))
    pu_indices = _pu_pool_for(hs_indices)
    print(f"[worker] HS files = {hs_indices}")
    print(f"[worker] PU pool  = {pu_indices}")
    print(f"[worker] chunk_size = {chunk_size}    event_name = {event_name}",
          flush=True)
    run_preprocessing_pipeline(
        r=hs_indices,
        pu_indices=pu_indices,
        chunk_size=chunk_size,
        event_name=event_name,
    )


def _run_manager(hs_start: int, hs_end: int, event_name: str,
                 chunk_size: int, group_size: int) -> None:
    """Split [hs_start, hs_end) into groups of `group_size`. For each group:
    spawn one subprocess with --worker, stream its stdout/stderr live."""
    n_total = hs_end - hs_start
    if n_total <= 0:
        print(f"empty HS range [{hs_start}, {hs_end}) — nothing to do")
        return

    starts = list(range(hs_start, hs_end, group_size))
    n_groups = len(starts)
    print(f"[manager] {n_total} HS files in [{hs_start}, {hs_end})  →  "
          f"{n_groups} groups of {group_size}")
    print(f"[manager] chunk_size={chunk_size}   event_name={event_name}")

    script = __file__
    overall_t0 = time.perf_counter()
    n_ok = n_fail = 0
    for gi, gs in enumerate(starts, 1):
        ge = min(gs + group_size, hs_end)
        hs_indices = list(range(gs, ge))
        pu_indices = _pu_pool_for(hs_indices)
        print(f"\n────────────────────────────────────────────────────────────────")
        print(f"[group {gi}/{n_groups}]  HS [{gs}, {ge}) = {hs_indices}  "
              f"→  PU {pu_indices}")
        print("────────────────────────────────────────────────────────────────",
              flush=True)
        cmd = [
            sys.executable, "-u", script,
            "--worker",
            "--hs-start", str(gs),
            "--hs-end",   str(ge),
            "--event-name", event_name,
            "--chunk-size", str(chunk_size),
        ]
        t0 = time.perf_counter()
        # Stream live — Popen with line-buffered stdout/stderr merged.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"\n  ! group {gi}/{n_groups} exited {rc} after {dt:.1f}s")
            n_fail += 1
        else:
            print(f"\n  ✓ group {gi}/{n_groups} done in {dt:.1f}s "
                  f"({dt/60:.2f} min)")
            n_ok += 1

    total_dt = time.perf_counter() - overall_t0
    print(f"\n[manager] all groups done: ok={n_ok} fail={n_fail} "
          f"total {total_dt:.1f}s ({total_dt/60:.2f} min)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hs-start", type=int, required=True)
    parser.add_argument("--hs-end",   type=int, required=True)
    parser.add_argument("--event-name", type=str, default="ttbar_pu0")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=3,
                        help="HS files per subprocess group (default: 3, "
                             "= one PU 3-file block per group)")
    parser.add_argument("--worker", action="store_true",
                        help="(internal) worker mode: process the given HS "
                             "range in-process and exit")
    args = parser.parse_args()

    if args.worker:
        _run_worker(args.hs_start, args.hs_end, args.event_name, args.chunk_size)
    else:
        _run_manager(args.hs_start, args.hs_end, args.event_name,
                     args.chunk_size, args.group_size)


if __name__ == "__main__":
    main()

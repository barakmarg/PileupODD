"""Batch submission: split a shard range into jobs and hand them to PBS.

Producing a full dataset means processing hundreds of shards, which is far more
than one walltime allowance. This module splits the configured shard range into
groups, and either runs them locally as one fresh subprocess per group or emits
the ``qsub`` commands to queue them.

One fresh process per group is the point, not an implementation detail. Polars'
allocator does not return memory to the OS within a process, so a long-lived
run creeps upward across shards until it is killed. Ending the process at every
group boundary makes the OS reclaim everything.

For overlay runs the default group size is 3, matching the pileup pool's 3-shard
blocks: the pool is loaded once per group, so aligning groups to blocks avoids
reloading it mid-group.

Replaces ``submit_preprocess_overlay_range.py`` and ``run_research.sh`` on
``master``, where the shard list was edited into the source of
``run_research_preprocess.py`` before each run.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from colliderml_pflow.config import Config

#: Default PBS resource request. Overridable from the command line.
DEFAULT_PBS_RESOURCES = "walltime=72:00:00,mem=40gb,ncpus=16,ngpus=1,io=0.1,gputype=A6000"


def group_shards(file_indices: List[int], group_size: int) -> List[List[int]]:
    """Split shard indices into consecutive groups of at most ``group_size``."""
    if group_size <= 0:
        return [list(file_indices)]
    return [list(file_indices[i:i + group_size])
            for i in range(0, len(file_indices), group_size)]


def _worker_command(config_path: Path, shards: List[int], extra_sets: List[str]) -> List[str]:
    """Build the CLI invocation that processes exactly ``shards``.

    The group's own ``dataset.file_indices`` override is appended *last* so it
    always wins: overrides are applied in order, and a user-supplied
    ``--set dataset.file_indices=...`` would otherwise cancel the split and
    make every job process the whole range.
    """
    cmd = [
        sys.executable, "-u", "-m", "colliderml_pflow",
        "preprocess",
        "--config", str(config_path),
    ]
    for item in extra_sets:
        cmd += ["--set", item]
    cmd += ["--set", f"dataset.file_indices={shards}"]
    return cmd


def run_groups_locally(
    cfg: Config,
    config_path: Path,
    group_size: int,
    extra_sets: Optional[List[str]] = None,
) -> int:
    """Process each group in its own subprocess, streaming output live.

    Args:
        cfg: the loaded configuration (used for the shard list and mode).
        config_path: the YAML file, re-read by each subprocess.
        group_size: shards per subprocess.
        extra_sets: additional ``key=value`` overrides passed through.

    Returns:
        Number of groups that failed.
    """
    groups = group_shards(cfg.dataset.resolved_file_indices(), group_size)
    print(f"[manager] {len(cfg.dataset.resolved_file_indices())} shard(s) -> "
          f"{len(groups)} group(s) of {group_size}")

    overall_t0 = time.perf_counter()
    n_ok = n_fail = 0
    for gi, shards in enumerate(groups, 1):
        print(f"\n{'-' * 68}")
        print(f"[group {gi}/{len(groups)}]  shards {shards}")
        print(f"{'-' * 68}", flush=True)

        cmd = _worker_command(config_path, shards, extra_sets or [])
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"\n  ! group {gi}/{len(groups)} exited {rc} after {dt:.1f}s")
            n_fail += 1
        else:
            print(f"\n  ok group {gi}/{len(groups)} done in {dt:.1f}s ({dt / 60:.2f} min)")
            n_ok += 1

    total = time.perf_counter() - overall_t0
    print(f"\n[manager] all groups done: ok={n_ok} fail={n_fail} "
          f"total {total:.1f}s ({total / 60:.2f} min)")
    return n_fail


def qsub_commands(
    cfg: Config,
    config_path: Path,
    group_size: int,
    job_name: str,
    resources: str = DEFAULT_PBS_RESOURCES,
    queue: str = "N",
    log_dir: str = "logs",
    extra_sets: Optional[List[str]] = None,
) -> List[str]:
    """Build one ``qsub`` command line per group.

    Each job runs the CLI directly via ``qsub -- <command>``, so no wrapper
    shell script has to be kept in sync with the configuration.

    Args:
        cfg: the loaded configuration.
        config_path: the YAML file each job re-reads.
        group_size: shards per job.
        job_name: base PBS job name; the group index is appended.
        resources: PBS ``-l`` resource string.
        queue: PBS queue name.
        log_dir: directory for stdout/stderr files.
        extra_sets: additional ``key=value`` overrides passed through.

    Returns:
        The command lines, ready to run or inspect.
    """
    groups = group_shards(cfg.dataset.resolved_file_indices(), group_size)
    cmds = []
    for gi, shards in enumerate(groups, 1):
        worker = " ".join(shlex.quote(p) for p in
                          _worker_command(config_path, shards, extra_sets or []))
        name = f"{job_name}-{gi:03d}"
        cmds.append(
            f"qsub -N {shlex.quote(name)} -q {shlex.quote(queue)} "
            f"-o {shlex.quote(f'{log_dir}/{name}.out')} "
            f"-e {shlex.quote(f'{log_dir}/{name}.err')} "
            f"-l {shlex.quote(resources)} -- {worker}"
        )
    return cmds


def submit_qsub(cmds: List[str], log_dir: str) -> None:
    """Run the prepared ``qsub`` command lines."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    for cmd in cmds:
        print(f"$ {cmd}", flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ! qsub failed: {result.stderr.strip()}")
        else:
            print(f"  submitted: {result.stdout.strip()}")

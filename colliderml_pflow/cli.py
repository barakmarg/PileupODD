"""Command-line entry point.

Three subcommands covering the path from raw HuggingFace shards to a
model-ready dataset::

    python -m colliderml_pflow preprocess --config configs/ttbar_pu200_all_vertices.yaml
    python -m colliderml_pflow norm-stats --config configs/ttbar_pu200_all_vertices.yaml
    python -m colliderml_pflow submit     --config configs/ttbar_pu200_all_vertices.yaml --dry-run

Train/validation/test splitting is deliberately absent: the dataset is written
as one flat set of shards, and the training dataloader does the splitting itself
by shuffling shard files at load time.

Any configuration value can be overridden without editing the YAML::

    ... preprocess --config configs/smoke.yaml --set mode=overlay --set runtime.chunk_size=2

Values are parsed as YAML, so ``null``, ``true`` and ``[0,1,2]`` all work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from colliderml_pflow.config import load_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file. Omit to use built-in defaults.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set runtime.chunk_size=25. "
                             "Repeatable.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colliderml_pflow",
        description="Build particle-flow training datasets from the ColliderML ODD sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Build the dataset from HuggingFace shards.")
    _add_common(p_pre)
    p_pre.add_argument("--print-config", action="store_true",
                       help="Print the resolved configuration and exit without running.")

    p_norm = sub.add_parser("norm-stats", help="Compute input normalization statistics.")
    _add_common(p_norm)
    p_norm.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset directory. Defaults to the config's output_dir.")
    p_norm.add_argument("--output", type=Path, default=None,
                        help="Destination YAML. Defaults to <data-dir>/normalization_stats.yaml.")
    p_norm.add_argument("--max-files", type=int, default=40,
                        help="Shards to scan; <=0 scans all (default: 40).")
    p_norm.add_argument("--kll-k", type=int, default=200,
                        help="KLL sketch accuracy parameter (default: 200).")

    p_sub = sub.add_parser("submit", help="Split the shard range into batch jobs.")
    _add_common(p_sub)
    p_sub.add_argument("--group-size", type=int, default=None,
                       help="Shards per job. Default: 3 for overlay (matching the pileup "
                            "pool blocks), 1 otherwise.")
    p_sub.add_argument("--local", action="store_true",
                       help="Run groups locally as sequential subprocesses instead of queueing.")
    p_sub.add_argument("--dry-run", action="store_true",
                       help="Print the qsub command lines without submitting.")
    p_sub.add_argument("--job-name", type=str, default=None,
                       help="Base PBS job name. Defaults to <event_name>-<mode>.")
    p_sub.add_argument("--queue", type=str, default="N", help="PBS queue (default: N).")
    p_sub.add_argument("--resources", type=str, default=None, help="PBS -l resource string.")
    p_sub.add_argument("--log-dir", type=str, default="logs", help="Job log directory.")

    return parser


def _cmd_preprocess(args) -> int:
    from colliderml_pflow.runner import run_preprocessing

    cfg = load_config(args.config, args.overrides)
    if args.print_config:
        print(cfg.describe())
        return 0
    run_preprocessing(cfg)
    return 0


def _cmd_norm_stats(args) -> int:
    from colliderml_pflow.normalization import write_normalization_stats

    cfg = load_config(args.config, args.overrides)
    data_dir = args.data_dir or cfg.resolved_output_dir()
    write_normalization_stats(data_dir, args.output,
                              max_files=args.max_files, kll_k=args.kll_k)
    return 0


def _cmd_submit(args) -> int:
    from colliderml_pflow.submit import (
        DEFAULT_PBS_RESOURCES, qsub_commands, run_groups_locally, submit_qsub,
    )

    cfg = load_config(args.config, args.overrides)
    if args.config is None:
        print("submit needs --config: each job re-reads the YAML file.", file=sys.stderr)
        return 2

    group_size = args.group_size
    if group_size is None:
        # Overlay loads its pileup pool once per group, and the pool is built
        # from 3-shard blocks, so groups of 3 avoid reloading it mid-group.
        group_size = 3 if cfg.is_overlay else 1

    if args.local:
        return 1 if run_groups_locally(cfg, args.config, group_size, args.overrides) else 0

    job_name = args.job_name or f"{cfg.dataset.event_name}-{cfg.mode}"
    cmds = qsub_commands(
        cfg, args.config, group_size, job_name,
        resources=args.resources or DEFAULT_PBS_RESOURCES,
        queue=args.queue, log_dir=args.log_dir, extra_sets=args.overrides,
    )
    if args.dry_run:
        print(f"# {len(cmds)} job(s) would be submitted:")
        for cmd in cmds:
            print(cmd)
        return 0
    submit_qsub(cmds, args.log_dir)
    return 0


_COMMANDS = {
    "preprocess": _cmd_preprocess,
    "norm-stats": _cmd_norm_stats,
    "submit": _cmd_submit,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

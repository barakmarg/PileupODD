#!/usr/bin/env python
"""
Quick runner script for multi-event CLUE optimizer.

Optimizes unified CLUE parameters (dc, rhoc, dm, ppbin) across multiple events.

Usage from terminal:
    python run_multi_event_optimizer.py --max-events 50 --n-trials 100
    python run_multi_event_optimizer.py --max-events 100 --n-trials 200 --num-files 2

Or import in notebook:
    from run_multi_event_optimizer import main
    best_params, study = main(max_events=100, n_trials=200)
"""

import argparse
import sys
import polars as pl
from huggingface_hub import HfFileSystem
from primary.multi_event_optimizer import run_multi_event_optimizer


def load_calo_hits(event_type='ttbar_pu200', num_files=1) -> pl.DataFrame:
    """Load calorimeter hits from HuggingFace Hub."""
    print(f"Loading {num_files} file(s) of {event_type} calorimeter hits...")

    fs = HfFileSystem()
    calo_hits_list = []

    for i in range(num_files):
        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_type}_calo_hits/train-{i:05d}-of-01000.parquet"
        print(f"  Loading: {file_path}")
        with fs.open(file_path, "rb") as f:
            calo_hits_list.append(pl.read_parquet(f))

    calo_hits = pl.concat(calo_hits_list)
    print(f"✓ Loaded {calo_hits.shape[0]} total hits from {calo_hits['event_id'].n_unique()} events")

    return calo_hits


def main(max_events: int = 50,
         n_trials: int = 100,
         event_type: str = 'ttbar_pu200',
         num_files: int = 1,
         seed: int = 42):
    """
    Main entry point for multi-event optimization.

    Args:
        max_events: Number of events to use for optimization
        n_trials: Number of Optuna trials
        event_type: Type of events (e.g., 'ttbar_pu200')
        num_files: Number of parquet files to load
        seed: Random seed

    Returns:
        (best_params_dict, optuna_study)
    """
    print("\n" + "="*70)
    print("MULTI-EVENT CLUE OPTIMIZER")
    print("Optimizes: dc, rhoc, dm, ppbin (unified across all events)")
    print("="*70 + "\n")

    # Load data
    calo_hits = load_calo_hits(event_type=event_type, num_files=num_files)

    # Run optimization
    best_params, study = run_multi_event_optimizer(
        calo_hits=calo_hits,
        max_events=max_events,
        n_trials=n_trials,
        seed=seed
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Event Type:      {event_type}")
    print(f"Max Events:      {max_events}")
    print(f"Trials:          {n_trials}")
    print(f"Best Objective:  {study.best_value:.6f}")
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key:10s}: {value}")

    return best_params, study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Event CLUE Optimizer (Unified Parameters)')
    parser.add_argument('--max-events', type=int, default=50,
                        help='Number of events to optimize on (default: 50)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of Optuna trials (default: 100)')
    parser.add_argument('--event-type', type=str, default='ttbar_pu200',
                        help='Event type (default: ttbar_pu200)')
    parser.add_argument('--num-files', type=int, default=1,
                        help='Number of parquet files to load (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    best_params, study = main(
        max_events=args.max_events,
        n_trials=args.n_trials,
        event_type=args.event_type,
        num_files=args.num_files,
        seed=args.seed
    )

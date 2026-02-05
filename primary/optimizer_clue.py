import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import optuna
from primary.calibration import CALIBRATION

# --------------------------------------------------------------------------------
# 1. SETUP & DEFINITIONS
# --------------------------------------------------------------------------------

def optimizer(calo_hits: pl.DataFrame, metric_eval: callable, execute: callable, n_trials=100):
    """
    Optimizes CLUE parameters using Optuna.
    
    Args:
        calo_hits: Polars DataFrame containing calorimeter hits.
        metric_eval: Function that takes a processed DataFrame and returns a loss metric (ratio).
        execute: Function that takes (calo_df, params_ecal, params_hcal) and returns (processed_df, total_clusters).
        n_trials: Number of trials for Optuna.
    """
    
    # Filter to single event for optimization speed
    calo = calo_hits.filter(pl.col('event_id') == 1)
    
    print(f"Starting Optuna Optimization with {n_trials} trials...")
    print("Note: Ensure 'execute' function accepts (calo, params_ecal, params_hcal) arguments.")

    def objective(trial):
        # 1. Sample Parameters using Optuna
        
        # ECAL Parameters
        # rhoc: Sensitive density threshold
        ecal_rhoc = trial.suggest_float("ecal_rhoc", 0.0, 0.30)
        # dc: Local density radius (mm)
        ecal_dc = trial.suggest_float("ecal_dc", 15.0, 100.0)
        # dm: Distance to nearest higher density. Defined as ratio to dc to ensure dm >= dc
        ecal_dm_ratio = trial.suggest_float("ecal_dm_ratio", 1.0, 1.5)
        
        current_ecal = {
            'dc': ecal_dc,
            'dm': ecal_dc * ecal_dm_ratio,
            'rhoc': ecal_rhoc,
            'max_neighbors': 400
        }
        
        # HCAL Parameters
        hcal_rhoc = trial.suggest_float("hcal_rhoc", 0.0, 0.60)
        hcal_dc = trial.suggest_float("hcal_dc", 40.0, 130.0)
        hcal_dm_ratio = trial.suggest_float("hcal_dm_ratio", 1.0, 1.5)

        current_hcal = {
            'dc': hcal_dc,
            'dm': hcal_dc * hcal_dm_ratio,
            'rhoc': hcal_rhoc,
            'max_neighbors': 400
        }
        
        try:
            # 2. Execute CLUE with sampled parameters
            calo_result, total_clusters = execute(calo, current_ecal, current_hcal)
            
            # 3. Evaluate Metric (Ratio of Noise/Signal or similar)
            ratio = metric_eval(calo_result)
            
            # 4. Objective Function Calculation
            # Soft constraints via penalty
            penalty = 0
            
            # Constraint: Limit total number of clusters to prevent fragmentation
            if total_clusters > 10000:
                penalty += (total_clusters - 10000) * 0.02
            
            # Constraint: Ratio should be reasonable
            if ratio > 1.0:
                penalty += (ratio - 1.0) * 10
            
            score = ratio + penalty
            
            # Record metrics for analysis
            trial.set_user_attr("clusters", total_clusters)
            trial.set_user_attr("ratio", ratio)
            
            return score
            
        except Exception as e:
            # Handle execution failures gracefully
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    # Create Study
    # TPE (Tree-structured Parzen Estimator) is generally efficient for hyperparameter tuning
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run Optimization
    # Show progress bar if optuna version supports it, or standard logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=n_trials)

    # --------------------------------------------------------------------------------
    # 3. RESULTS & FINAL RUN
    # --------------------------------------------------------------------------------
    print("\n--- Optimization Finished ---")
    print(f"Best Objective Value: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Reconstruct Best Parameter Dictionaries
    bp = study.best_params
    best_ecal_params = {
        'dc': bp['ecal_dc'],
        'dm': bp['ecal_dc'] * bp['ecal_dm_ratio'],
        'rhoc': bp['ecal_rhoc'],
        'max_neighbors': 300
    }
    best_hcal_params = {
        'dc': bp['hcal_dc'],
        'dm': bp['hcal_dc'] * bp['hcal_dm_ratio'],
        'rhoc': bp['hcal_rhoc'],
        'max_neighbors': 300
    }
    
    print("\n--- Running Final Execution with Best Parameters ---")
    final_calo, final_total_clusters = execute(calo, best_ecal_params, best_hcal_params)
    final_ratio = metric_eval(final_calo)

    print(f"\nFinal Result:")
    print(f"Total Clusters: {final_total_clusters}") 
    print(f"Ratio: {final_ratio:.4f}")
    
    return final_calo, study
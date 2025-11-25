import numpy as np
from datasets.arrow_dataset import Dataset
from typing import Tuple, List

import polars as pl


# --- 2. C-BACKEND HIGH PERFORMANCE CALCULATION ---
def compute_masks_vectorized(df: pl.DataFrame):
    """
    Calculates calorimeter masks using the Rust/C backend.
    Faster than Numpy loops by orders of magnitude.
    """
    print("Starting vectorized calculation...")
    
    # We assume 'vx', 'vy', 'vz' are columns containing Lists (Jagged Arrays).
    # To make this run at C-speed, we avoid looping over lists.
    # Strategy: Explode (Flatten) -> Calc -> Group (Restructure)
    
    result = (
        df.lazy()
        .with_row_index("event_idx") # Keep track of event ID
        .select([
            pl.col("event_idx"),
            pl.col("vx"),
            pl.col("vy"), 
            pl.col("vz")
        ])
        # 1. FLATTEN: Turns jagged lists into one contiguous memory block.
        #    This allows the CPU to process millions of particles without pointers.
        .explode(["vx", "vy", "vz"])
        .with_columns(
            # 2. COMPUTE: The actual math (runs in Rust/C++)
            (
                ((pl.col("vx").pow(2) + pl.col("vy").pow(2)).sqrt() > 1400)
                | 
                (pl.col("vz").abs() > 3000)
            ).alias("mask")
        )
        # 3. RESTRUCTURE: Pack the booleans back into lists per event
        .group_by("event_idx", maintain_order=True)
        .agg(pl.col("mask"))
        .collect() # Trigger execution
    )
    
    return result["mask"]
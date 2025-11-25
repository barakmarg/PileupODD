import polars as pl

def cast_parent_id_to_int64(df: pl.DataFrame) -> pl.DataFrame:
    """
    Casts the 'parent_id' column from List<Float> to List<Int64>.
    
    This is a Zero-Copy operation where possible, but if the data 
    was physically Float on disk, it creates a new Integer array in RAM.
    """
    return df.with_columns(
        # We specify the target type as a List containing Int64s
        pl.col("parent_id").cast(pl.List(pl.Int64))
    ) 

def add_orphan_mask(df: pl.DataFrame) -> pl.DataFrame:
    print("Computing parent existence mask...")

    df_indexed = df.with_row_index("tmp_event_idx")

    # 1. Build Lookup Table
    valid_ids_lookup = (
        df_indexed.select(["tmp_event_idx", "particle_id"])
        .explode("particle_id")
        .rename({"particle_id": "valid_pid"})
        # Ensure ID types match (Int64 vs Int64)
        .with_columns(pl.col("valid_pid").cast(pl.Int64))
        .unique()
        # --- THE FIX: Add a tracer column ---
        # We need this because 'valid_pid' gets dropped during the join.
        .with_columns(pl.lit(True).alias("found_in_event")) 
    )

    # 2. Flatten Parent IDs
    # (Assuming you already ran cast_parent_id_to_int64, so parent_id is Int64)
    parents_flat = (
        df_indexed.select(["tmp_event_idx", "parent_id"])
        .explode("parent_id")
        .with_row_index("original_order")
    )

    # 3. Join
    matched = parents_flat.join(
        valid_ids_lookup,
        left_on=["tmp_event_idx", "parent_id"],
        right_on=["tmp_event_idx", "valid_pid"],
        how="left"
    )

    # 4. Check the Tracer
    result_mask = (
        matched
        .sort("original_order")
        .with_columns(
            # If 'found_in_event' is Null, the join failed -> Parent Missing
            pl.col("found_in_event").is_null().alias("is_parent_missing")
        )
        .group_by("tmp_event_idx", maintain_order=True)
        .agg(pl.col("is_parent_missing"))
    )

    # 5. Merge back
    return (
        df_indexed
        .join(result_mask, on="tmp_event_idx", how="left")
        .drop("tmp_event_idx")
    )
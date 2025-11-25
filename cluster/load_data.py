import os
import io
import pyarrow as pa
from datasets.arrow_dataset import Dataset

import os
import pyarrow as pa
import polars as pl

# --- 1. HDD-OPTIMIZED LOADING ---
def load_dataset_to_ram_optimized(cache_files_list):
    """
    Loads all Arrow files into a single Polars DataFrame with minimal RAM overhead.
    """
    chunks = []
    
    # 8MB buffer tells the HDD to read long sequential tracks
    # reducing the physical movement of the disk head.
    HDD_BUFFER = 8 * 1024 * 1024 

    for file_info in cache_files_list:
        file_path = file_info['filename']
        if not os.path.exists(file_path):
            continue

        print(f"Loading: {os.path.basename(file_path)}")

        # optimization: buffering=HDD_BUFFER forces sequential disk reads
        with open(file_path, "rb", buffering=HDD_BUFFER) as f:
            try:
                # VITAL: We pass the file object 'f' directly.
                # We do NOT use f.read(). This saves 50% RAM immediately.
                reader = pa.ipc.open_stream(f)
            except pa.lib.ArrowInvalid:
                f.seek(0)
                reader = pa.ipc.open_file(f)
            
            # Read into Arrow Table
            arrow_table = reader.read_all()
            
            # Zero-Copy conversion to Polars (Instant, no extra RAM used)
            chunks.append(pl.from_arrow(arrow_table))

    if not chunks:
        return None

    print("Concatenating in memory...")
    # Combine all chunks into one massive DataFrame
    return pl.concat(chunks)



# --- USAGE ---

# 1. Load Data (Efficiently)
# files = [{'filename': 'data.arrow'}]
# df = load_dataset_to_ram_optimized(files)

# 2. Calculate (Fast)
# result_masks = compute_masks_vectorized(df)

# 'result_masks' is now a Polars Series containing Lists of Booleans.
# If you strictly need a list of numpy arrays (warning: this is slow/heavy):
# final_numpy_list = result_masks.to_list()
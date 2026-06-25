"""Run the all-vertices chunked research preprocessing pipeline -> paper_optuna.

Same pipeline as run_research_preprocess.py, but processes a fresh set of 100
HuggingFace file indices (100 events each => 10,000 events) that do NOT overlap
with the existing ttbar_pu200_all_vertices_paper dataset, and writes outputs to
.../ttbar_pu200_all_vertices_paper_optuna.

Edit FILE_INDICES / EVENT_NAME / CHUNK_SIZE / OUTPUT_DIR as needed.
"""
import sys

sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

from create_trainning_dataset_pileup_research import (
    run_preprocessing_pipeline_all_vertices_chunked,
)

# --- configuration ---------------------------------------------------------
EVENT_NAME = "ttbar_pu200"
CHUNK_SIZE = 100

# 100 file indices (100 events each => 10,000 events). These are the first 100
# indices in 0..999 that are NOT already used by ttbar_pu200_all_vertices_paper,
# so the two datasets are disjoint.
FILE_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 83,
    84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 97, 98, 99, 101, 102, 103, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 116,
]

# Output dir name carries the "paper_optuna" tag the user asked for.
OUTPUT_DIR = f"/storage/agrp/barakma/PileupODD/data/{EVENT_NAME}_all_vertices_paper_optuna"
# ---------------------------------------------------------------------------


def main() -> None:
    # Optional positional args slice into FILE_INDICES (a range over the list,
    # NOT over file indices). With no args, the whole FILE_INDICES list runs.
    start = int(sys.argv[1]) if len(sys.argv) > 1 else None
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    selected = FILE_INDICES[start:end]

    print(
        f"Processing {len(selected)} file indices "
        f"(FILE_INDICES[{start}:{end}]): {selected}"
    )
    run_preprocessing_pipeline_all_vertices_chunked(
        r=selected,
        event_name=EVENT_NAME,
        chunk_size=CHUNK_SIZE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()

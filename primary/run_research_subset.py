"""Run the research pipeline on a fixed subset of file indices.

This script extracts indices from a list of target_particles parquet paths and
runs the all-vertices chunked pipeline only for those indices.
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

from pathlib import Path

from create_trainning_dataset_pileup_research import (
    run_preprocessing_pipeline_all_vertices_chunked,
)


VAL_FILES = [
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00686.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00214.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00363.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00379.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00166.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00373.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00854.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00650.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00464.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00927.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00920.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00103.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00951.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00875.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00081.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00296.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00791.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00233.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00677.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00046.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00071.parquet",
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200/target_particles-00721.parquet",
]


def _extract_index(path_str: str) -> int:
    name = Path(path_str).name
    stem = name.split(".")[0]
    idx_str = stem.split("-")[-1]
    return int(idx_str)


def main() -> None:
    #indices = sorted({ _extract_index(p) for p in VAL_FILES })
    run_preprocessing_pipeline_all_vertices_chunked(
        #r=indices,
        event_name="dihiggs_pu200",
        r=range(10,20),  # Run on all files, but only extract the specified indices
        chunk_size=100,
    )


if __name__ == "__main__":
    main()

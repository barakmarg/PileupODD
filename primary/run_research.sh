#!/bin/bash
# run_research.sh — runs the all-vertices chunked research preprocessing pipeline.
# File indices, event name and chunk size are configured inside
# run_research_preprocess.py (edit FILE_INDICES there).
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/PileupODD/primary
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Loop over positions in FILE_INDICES, launching ONE fresh python process per
# index. A new process per file index means the OS fully reclaims its memory
# between files (polars/mimalloc arenas don't accumulate across indices).
#
# START..END are positions into FILE_INDICES (END exclusive). Override via args:
#   ./run_research.sh 0 6   -> process FILE_INDICES[0], [1], ..., [5]
START=${1:-2}
END=${2:-100}

for ((idx=START; idx<END; idx++)); do
    echo "=== [run_research.sh] Processing FILE_INDICES position ${idx} ==="
    python run_research_preprocess.py "${idx}" "$((idx+1))"
done
popd

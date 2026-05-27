#!/bin/bash
# Body of the qsub job for H→bb̄ jet analysis.
# The script reads the bb̄ events from
#   /storage/agrp/barakma/PileupODD/data/hf_decay_labels/labels_file_*.parquet
# so it does not need a file range — it decides which events to process
# based on the labels. Each chunk runs in its own subprocess for memory
# isolation; OS reclaims polars cache on every chunk boundary.

export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/PileupODD/analysis

N_BB_EVENTS="${N_BB_EVENTS:-300}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"

echo "inspect_hbb_top_bottoms: n_bb_events=${N_BB_EVENTS}, chunk_size=${CHUNK_SIZE}"
python -u inspect_hbb_top_bottoms.py \
    --n-bb-events "${N_BB_EVENTS}" \
    --chunk-size  "${CHUNK_SIZE}"

popd

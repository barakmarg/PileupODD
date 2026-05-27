#!/bin/bash
# Body of the qsub job for HF Higgs-decay channel classification.
# Mirrors the convention used by primary/run.sh.
#
# Expects RANGE_START, RANGE_END (inclusive..exclusive, like Python slicing)
# passed in by the qsub -v switch in submit_classify.sh.

export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/PileupODD/analysis

# Build the explicit file list [RANGE_START .. RANGE_END-1]

echo "classifying HF files ${RANGE_START}..$((RANGE_END - 1))"
python -u classify_hf_decay_channels.py 

popd

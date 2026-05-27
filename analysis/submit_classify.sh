#!/bin/bash
# Submit the HF Higgs-decay channel classifier as a qsub job.
# Mirrors primary/brk.sh.
#
# The classifier processes HF files [RANGE_START, RANGE_END), spawning one
# subprocess per file. Labels land in
#   /storage/agrp/barakma/PileupODD/data/hf_decay_labels/labels_file_NNNNN.parquet
# Resumable: re-running with the same range skips files whose parquet exists.

export IOTHROTTLE_LIMIT=100
pushd /storage/agrp/barakma/PileupODD/analysis

# Default: classify all 1000 HF files. Override on the command line, e.g.
#   RANGE_START=200 RANGE_END=400 ./submit_classify.sh
RANGE_START="${RANGE_START:-0}"
RANGE_END="${RANGE_END:-1000}"

qsub -v RANGE_START=$RANGE_START,RANGE_END=$RANGE_END \
     -o classify_output.log \
     -e classify_error.log \
     -q N \
     -N hf_classify \
     -l walltime=72:00:00,mem=40gb,ncpus=4,io=0.1 \
     run_classify.sh

popd

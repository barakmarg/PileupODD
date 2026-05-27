#!/bin/bash
# Submit the H→bb̄ jet-analysis job. Picks bb̄ events directly from the
# labels parquets in /storage/agrp/barakma/PileupODD/data/hf_decay_labels/
# (written by classify_hf_decay_channels.py) — no file range needed.
#
# Override via env vars, e.g.:
#   N_BB_EVENTS=500 CHUNK_SIZE=50 ./submit_inspect_hbb.sh

export IOTHROTTLE_LIMIT=100
pushd /storage/agrp/barakma/PileupODD/analysis

N_BB_EVENTS="${N_BB_EVENTS:-300}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"

qsub -v N_BB_EVENTS=$N_BB_EVENTS,CHUNK_SIZE=$CHUNK_SIZE \
     -o inspect_hbb_output.log \
     -e inspect_hbb_error.log \
     -q N \
     -N inspect_hbb \
     -l walltime=72:00:00,mem=40gb,ncpus=4,io=0.1 \
     run_inspect_hbb.sh

popd

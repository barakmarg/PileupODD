#!/bin/bash
# run_research_optuna.sh — all-vertices chunked research preprocessing for the
# paper_optuna dataset (10,000 fresh events -> ttbar_pu200_all_vertices_paper_optuna).
# File indices, output dir, event name and chunk size are configured inside
# run_research_optuna_preprocess.py.
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/PileupODD/primary
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Loop over positions in FILE_INDICES, launching ONE fresh python process per
# index so the OS reclaims memory between files.
#
# START..END are positions into FILE_INDICES (END exclusive). Override via args:
#   ./run_research_optuna.sh 0 6   -> process FILE_INDICES[0], [1], ..., [5]
START=${1:-0}
END=${2:-100}
#qsub -o output_optuna.log -e error_optuna.log -q N -N ttbar-allvert-optuna -l walltime=72:00:00,mem=40gb,ncpus=16,ngpus=1,io=0.1,gputype=A6000 run_research_optuna.sh

for ((idx=START; idx<END; idx++)); do
    echo "=== [run_research_optuna.sh] Processing FILE_INDICES position ${idx} ==="
    python run_research_optuna_preprocess.py "${idx}" "$((idx+1))"
done
popd

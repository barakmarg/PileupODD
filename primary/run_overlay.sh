# run_overlay.sh — qsub job body for the PU overlay pipeline.
# Expects HS_START, HS_END, EVENT_NAME, and (optionally) CHUNK_SIZE,
# GROUP_SIZE in the env (passed by submit_overlay.sh via qsub -v).
#
# Calls submit_preprocess_overlay_range.py in MANAGER mode: it splits the
# HS range into 3-file groups, spawns one subprocess per group, and
# streams each subprocess's stdout to this job's output log.
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/PileupODD/primary
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHUNK_SIZE="${CHUNK_SIZE:-334}"
GROUP_SIZE="${GROUP_SIZE:-3}"

echo "running overlay: HS [${HS_START}, ${HS_END})  EVENT_NAME=${EVENT_NAME}  "\
"CHUNK_SIZE=${CHUNK_SIZE}  GROUP_SIZE=${GROUP_SIZE}"

python -u submit_preprocess_overlay_range.py \
    --hs-start    "${HS_START}" \
    --hs-end      "${HS_END}" \
    --event-name  "${EVENT_NAME}" \
    --chunk-size  "${CHUNK_SIZE}" \
    --group-size  "${GROUP_SIZE}"

popd

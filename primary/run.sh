# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/PileupODD/primary
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# send three arguments: range_start, range_end, event_name
python submit_preprocess.py $RANGE_START $RANGE_END $EVENT_NAME
popd

# submit_overlay.sh — qsub launcher for the PU overlay pipeline.
# Mirrors submit.sh, but targets the overlay (manager+worker) script.
#
# The overlay range [HS_START, HS_END) is split into 3-file groups inside
# the manager Python script; each group runs as its own subprocess so
# polars/GPU/temp memory is released between groups. Override via env
# vars on the cmd line, e.g.:
#     HS_START=0 HS_END=30 EVENT_NAME=ttbar_pu0 ./submit_overlay.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
pushd /storage/agrp/barakma/PileupODD/primary

# Defaults (override on the command line)
HS_START="${HS_START:-706}"
HS_END="${HS_END:-999}"
EVENT_NAME="${EVENT_NAME:-ttbar_pu0}"
CHUNK_SIZE="${CHUNK_SIZE:-334}"
GROUP_SIZE="${GROUP_SIZE:-3}"

qsub \
    -v HS_START=$HS_START,HS_END=$HS_END,EVENT_NAME=$EVENT_NAME,CHUNK_SIZE=$CHUNK_SIZE,GROUP_SIZE=$GROUP_SIZE \
    -o overlay_output.log \
    -e overlay_error.log \
    -q N \
    -N overlay \
    -l walltime=72:00:00,mem=300gb,ncpus=16,ngpus=1,io=0.1,gputype=A6000 \
    run_overlay.sh

popd

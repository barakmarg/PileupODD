# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
pushd /storage/agrp/barakma/PileupODD/primary

# Define parameters
RANGE_START=15
RANGE_END=25
EVENT_NAME=ttbar_pu0

qsub -v RANGE_START=$RANGE_START,RANGE_END=$RANGE_END,EVENT_NAME=$EVENT_NAME -o output.log -e error.log -q N -N progress -l walltime=72:00:00,mem=40gb,ncpus=16,ngpus=0,io=10.0 run.sh

popd
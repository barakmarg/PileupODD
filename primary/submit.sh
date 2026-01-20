# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
pushd /storage/agrp/barakma/PileupODD/primary



qsub -o output.log -e error.log -q N -N progress -l walltime=72:00:00,mem=48gb,ncpus=32,ngpus=0,io=10.0,gputype=A5000 run.sh

popd
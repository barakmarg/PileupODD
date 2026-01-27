# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export IOTHROTTLE_LIMIT=100
pushd /storage/agrp/barakma/PileupODD/primary



qsub -o output.log -e error.log -q N -N progress -l walltime=72:00:00,mem=16gb,ncpus=1,ngpus=0,io=0.10 run.sh

popd
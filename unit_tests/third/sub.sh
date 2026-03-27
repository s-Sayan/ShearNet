#!/bin/bash

# Submit training job and get its ID
JOBID_TRAIN=$(sbatch ./training/train_sn.sh | awk '{print $4}')
echo "Submitted training job: $JOBID_TRAIN"

# Submit m job, after training completes
JOBID_M=$(sbatch --dependency=afterok:$JOBID_TRAIN ./benchmarking/m/run_mcal.sh | awk '{print $4}')
echo "Submitted m job: $JOBID_M (depends on $JOBID_TRAIN)"

# Submit leakage job, after training completes
JOBID_PSF=$(sbatch --dependency=afterok:$JOBID_TRAIN ./benchmarking/psf_leakage/run_leakage.sh | awk '{print $4}')
echo "Submitted leakage job: $JOBID_PSF (depends on $JOBID_TRAIN)"
#!/bin/bash

# Submit array job (0-20)
ARRAY_JOBID=$(sbatch run_all_lpp_array.sh | awk '{print $4}')

echo "Submitted array job with ID: $ARRAY_JOBID"

# Submit merge job, only AFTER array completes
sbatch --dependency=afterok:${ARRAY_JOBID} merge_results.sh

echo "Submitted merge job dependent on array job $ARRAY_JOBID"

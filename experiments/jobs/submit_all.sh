#!/bin/bash


# Submit array job (0-20)
ARRAY_JOBID=$(sbatch run_all_lpp_array.sh | awk '{print $4}')

echo "Submitted array job with ID: $ARRAY_JOBID"

# Submit merge job, only AFTER array completes
sbatch --dependency=afterok:${ARRAY_JOBID} merge_results.sh

echo "Submitted merge job dependent on array job $ARRAY_JOBID"
# -------------------------------------------------------
# 0. Define root directory of your project (ABSOLUTE PATH)
# -------------------------------------------------------
export ROOT_DIR=/home/zahrab98/projects/aip-krallen/zahrab98/programmatic-policy-learning

# -------------------------------------------------------
# 1. Export OpenAI key for all child jobs
# -------------------------------------------------------
export OPENAI_API_KEY="${OPENAI_API_KEY}"

# -------------------------------------------------------
# 2. Generate experiment timestamp (shared by all jobs)
# -------------------------------------------------------
TS=$(date +"%Y-%m-%d/%H-%M-%S")
export EXPERIMENT_TS=$TS

echo "Launching LPP experiment with timestamp: $EXPERIMENT_TS"

# -------------------------------------------------------
# 3. Create REQUIRED folders BEFORE submitting SLURM jobs
#    (SLURM NEVER creates parent directories)
# -------------------------------------------------------
mkdir -p ${ROOT_DIR}/logs/${EXPERIMENT_TS}/slurm
mkdir -p ${ROOT_DIR}/logs/${EXPERIMENT_TS}/merge

# -------------------------------------------------------
# 4. Submit the job array (21 tasks)
# -------------------------------------------------------
ARRAY_JOBID=$(sbatch \
    --export=ALL,ROOT_DIR,EXPERIMENT_TS,OPENAI_API_KEY \
    ${ROOT_DIR}/experiments/jobs/run_all_lpp_array.sh \
    | awk '{print $4}')

echo "Submitted array job with ID: $ARRAY_JOBID"

# -------------------------------------------------------
# 5. Submit merge job, AFTER array finishes
# -------------------------------------------------------
sbatch \
    --dependency=afterok:${ARRAY_JOBID} \
    --export=ALL,ROOT_DIR,EXPERIMENT_TS,OPENAI_API_KEY \
    ${ROOT_DIR}/experiments/jobs/merge_results.sh

echo "Submitted merge job with dependency on array job $ARRAY_JOBID"

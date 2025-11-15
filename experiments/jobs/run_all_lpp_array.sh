#!/bin/bash
#SBATCH --job-name=lpp_array
#SBATCH --account=aip-krallen
#SBATCH --array=0-20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

module load StdEnv/2023
module load python/3.11

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate
cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning

#############################################
# Determine DSL variant and SEED for this job
#############################################

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    DSL_NAME="base_prime"
    SEED=0
else
    DSL_NAME="llm"
    SEEDS=({0..19})
    INDEX=$((SLURM_ARRAY_TASK_ID - 1))
    SEED=${SEEDS[$INDEX]}
fi

echo "Task $SLURM_ARRAY_TASK_ID running DSL=$DSL_NAME seed=$SEED"

#############################################
# Run the Python evaluation
#############################################

python -u experiments/run_experiment.py \
    dsl_name=${DSL_NAME} \
    seed=${SEED} \
    hydra.run.dir=outputs/${DSL_NAME}_${SEED}

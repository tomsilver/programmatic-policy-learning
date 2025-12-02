#!/bin/bash
#SBATCH --job-name=lpp_array
#SBATCH --account=aip-krallen
#SBATCH --array=0-20
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=slurm/array_%A_%a.out
#SBATCH --error=slurm/array_%A_%a.err


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
=======
=======
#############################
# List of envs for this experiment
#############################

ENVS=("ggg_nim" "ggg_chase")   # change here if you add/remove envs

#############################
# Run each env sequentially, with its own dir
#############################

# python -u experiments/run_experiment.py -m\
#     env=ggg_nim,ggg_chase,ggg_checkmate,ggg_rfts,ggg_stf \
#     approach=lpp \
#     dsl_name=${DSL_NAME} \
#     seed=${SEED} \
#     approach.demo_numbers='[0,1,2,3,4,5,6,7,8,9,10]','[0,1,2,3,4,5]'
    # hydra.run.dir=${RUN_DIR}
for ENV in "${ENVS[@]}"; do
    RUN_DIR="${ROOT_DIR}/logs/${EXPERIMENT_TS}/${ENV}/${DSL_NAME}_${SEED}"
    mkdir -p "$RUN_DIR"

    echo "Running env=${ENV} into ${RUN_DIR}"

    python -u experiments/run_experiment.py \
        env=${ENV} \
        approach=lpp \
        dsl_name=${DSL_NAME} \
        seed=${SEED} \
	hydra.run.dir=${RUN_DIR}
done

=======


#############################
# Alternative multirun
#############################

# python -u experiments/run_experiment.py -m\
#     env=ggg_nim,ggg_chase,ggg_checkmate,ggg_rfts,ggg_stf \
#     approach=lpp \
#     dsl_name=${DSL_NAME} \
#     seed=${SEED} \
#     approach.demo_numbers='[0,1,2,3,4,5,6,7,8,9,10]','[0,1,2,3,4,5]'

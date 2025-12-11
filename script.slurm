#!/bin/bash
#SBATCH --job-name=rfts_sleep
#SBATCH --account=aip-krallen
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --output=rfts_sleep/%x-%j.out
#SBATCH --error=rfts_sleep/%x-%j.out


module load StdEnv/2023
module load python/3.11

export OPENAI_API_KEY="${OPENAI_API_KEY}"

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate
cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning

echo "Running experiment:"
echo "  ENV_NAME = $ENV_NAME"
echo "  SEED     = $SEED"
echo "  PROMPT_FILE = $PROMPT_FILE"
echo "  NUM_PROGRAMS = $NUM_PROGRAMS"
echo "  STEP_SIZE = $STEP_SIZE"
echo "  NUM_DTS = $NUM_DTS"
echo "  NUM_PARTICLES = $NUM_PARTICLES"


python -u experiments/run_experiment.py \
    env=$ENV_NAME \
    approach=lpp \
    seed=$SEED \
    approach.program_generation.dsl_generator_prompt=src/programmatic_policy_learning/dsl/llm_primitives/prompts/full/$PROMPT_FILE\
    approach.num_programs=$NUM_PROGRAMS\
    approach.program_generation_step_size=$STEP_SIZE\
    approach.num_dts=$NUM_DTS\
    approach.max_num_particles=$NUM_PARTICLES\

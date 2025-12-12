#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --account=aip-krallen
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out


module load StdEnv/2023
module load python/3.11

export OPENAI_API_KEY="${OPENAI_API_KEY}"

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate
cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning



python -u experiments/run_experiment.py \
    env=ggg_nim \
    approach=lpp \
    seed=0 \
    approach.program_generation.dsl_generator_prompt=src/programmatic_policy_learning/dsl/llm_primitives/prompts/full/v3_nim.txt\
    approach.num_programs=200\
    approach.program_generation_step_size=1\
    approach.num_dts=5\
    approach.max_num_particles=25\

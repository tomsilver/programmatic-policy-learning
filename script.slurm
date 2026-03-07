#!/bin/bash
#SBATCH --job-name=test_stf
#SBATCH --account=aip-krallen
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out


module load StdEnv/2023
module load python/3.11

export OPENAI_API_KEY="${OPENAI_API_KEY}"
echo "Starting script..."

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate
cd ~/projects/aip-krallen/zahrab98/baselines/programmatic-policy-learning

echo "now..."

python3 hello_world.py
# python -u experiments/run_experiment.py\
#     env=ggg_stf \
#     approach=lpp \
#     seed=0 \
#     approach.program_generation.py_feature_gen_prompt=src/programmatic_policy_learning/dsl/llm_primitives/prompts/py_feature_gen/prompt_v1_0.txt\
#     approach.program_generation.py_feature_gen_batch_prompt=src/programmatic_policy_learning/dsl/llm_primitives/prompts/py_feature_gen/batch_prompt_v0.txt\
#     approach.program_generation.strategy=py_feature_gen \
#     approach.program_generation.num_batches=10\
#     approach.program_generation.num_features=1000\
#     approach.program_generation.encoding_method="4" \
#     approach.program_generation.hint_structured= True \
#     approach.num_programs=10000\
#     approach.program_generation_step_size=10\
#     approach.num_dts=5\
#     approach.max_num_particles=25

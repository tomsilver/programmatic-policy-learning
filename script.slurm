#!/bin/bash
#SBATCH --job-name=chase_at_cell_11_1k_100       # Job name
#SBATCH --account=aip-krallen       # Your advisor's account
#SBATCH --time=18:00:00              # Walltime (hh:mm:ss)
#SBATCH --mem=16G                    # Memory
#SBATCH --cpus-per-task=1           # Number of CPUs
#SBATCH --gres=gpu:0                # Request 0 GPUs; set to gpu:1 if you need a GPU
#SBATCH --output=%x-%j.out          # Output file (%x=job name, %j=job ID)
#SBATCH --error=%x-%j.out

# --- Load environment modules ---
module load StdEnv/2023
module load python/3.11

export OPENAI_API_KEY="${OPENAI_API_KEY}"

# --- Activate your virtual environment ---
source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate

# --- Move to your project directory ---
cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning

# --- Run your code ---
python -u experiments/run_experiment.py -m env=ggg_chase approach=lpp


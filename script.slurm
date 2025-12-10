#!/bin/bash
#SBATCH --job-name=check_50k_1k_dt20_par100      # Job name
#SBATCH --account=aip-krallen       # Your advisor's account
#SBATCH --time=36:00:00              # Walltime (hh:mm:ss)
#SBATCH --mem=16G                    # Memory
#SBATCH --cpus-per-task=1           # Number of CPUs
#SBATCH --gres=gpu:0                # Request 0 GPUs; set to gpu:1 if you need a GPU
#SBATCH --output=checkmate_v3/%x-%j.out          # Output file (%x=job name, %j=job ID)
#SBATCH --error=checkmate_v3/%x-%j.out

# --- Load environment modules ---
module load StdEnv/2023
module load python/3.11

export OPENAI_API_KEY="${OPENAI_API_KEY}"

# --- Activate your virtual environment ---
source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate

# --- Move to your project directory ---
cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning

# --- Run your code ---
python -u experiments/run_experiment.py -m env=ggg_checkmate approach=lpp


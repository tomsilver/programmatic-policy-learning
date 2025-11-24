#!/bin/bash
#SBATCH --job-name=lpp_merge
#SBATCH --account=aip-krallen
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=merge/merge_%j.out
#SBATCH --error=merge/merge_%j.err

module load StdEnv/2023
module load python/3.11

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate
cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning

if [[ -z "$EXPERIMENT_TS" ]]; then
    echo "ERROR: EXPERIMENT_TS is not set!"
    exit 1
fi

mkdir -p logs/${EXPERIMENT_TS}/merge

echo "Running merge job for experiment timestamp: $EXPERIMENT_TS"

python experiments/merge_results.py
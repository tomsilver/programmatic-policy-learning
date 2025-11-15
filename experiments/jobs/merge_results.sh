#!/bin/bash
#SBATCH --job-name=merge_results
#SBATCH --account=aip-krallen
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=merge-%j.out

module load StdEnv/2023
module load python/3.11

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate

cd ~/projects/aip-krallen/zahrab98/programmatic-policy-learning

python merge_results.py
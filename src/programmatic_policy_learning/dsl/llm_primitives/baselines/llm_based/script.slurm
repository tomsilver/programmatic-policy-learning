#!/bin/bash
#SBATCH --job-name=cap_baseline
#SBATCH --account=aip-krallen
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out


module load StdEnv/2023
module load python/3.11

export OPENAI_API_KEY="${OPENAI_API_KEY}"

source ~/projects/aip-krallen/zahrab98/programmatic-policy-learning/.venv/bin/activate

for ENV in ReachForTheStar TwoPileNim StopTheFall Chase CheckmateTactic; do
  python3 CaP_baseline.py --env "$ENV" --seeds 0 1 2 3 4 5 6 7 8 9 --encodings 1 2 3 4 --num-initial-states 4 --plot-results
done
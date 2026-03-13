import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Load all expert summaries
# -----------------------------
summary_files = glob.glob("logs/pendulum_expert_seed*_phased_summary.csv")
assert len(summary_files) > 0, "No expert summary files found."

dfs = [pd.read_csv(f) for f in summary_files]
all_data = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Aggregate across seeds
# -----------------------------
agg = (
    all_data.groupby(["phase", "timesteps"])
    .agg(
        mean_reward=("mean_reward", "mean"),
        std_reward=("mean_reward", "std"),
    )
    .reset_index()
)

# -----------------------------
# Plot
# -----------------------------
x = agg["timesteps"].values
y = agg["mean_reward"].values
yerr = agg["std_reward"].values

plt.figure(figsize=(7, 5))
plt.plot(x, y, marker="o", label="Hand-written prior + residual (TD3)")
plt.fill_between(x, y - yerr, y + yerr, alpha=0.25)

plt.xlabel("Cumulative training timesteps (residual)")
plt.ylabel("Total reward (mean over seeds & episodes)")
plt.title("Residual learning on hand-written pendulum controller (multi-seed)")
plt.legend()
plt.tight_layout()

plt.savefig("handwritten_residual_multiseed.png", dpi=200)
plt.close()

print("Saved figure to handwritten_residual_multiseed.png")

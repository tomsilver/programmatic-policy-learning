import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_variant(variant: str) -> pd.DataFrame:
    paths = sorted(glob.glob(f"logs/pendulum_{variant}_seed*_phased_summary.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No summary CSVs found for variant={variant}. "
            f"Expected files like logs/pendulum_{variant}_seedXXXX_phased_summary.csv"
        )
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df["variant"] = variant
    return df


def agg_across_seeds(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # value_col is already a per-seed mean over episodes; now aggregate across seeds
    agg = (
        df.groupby(["variant", "phase", "timesteps"], as_index=False)
        .agg(mean=(value_col, "mean"), std=(value_col, "std"))
        .sort_values(["variant", "timesteps"])
    )
    return agg


def plot_two_curves(agg: pd.DataFrame, title: str, ylabel: str, outpath: str):
    plt.figure(figsize=(7, 5))
    for variant, sub in agg.groupby("variant"):
        x = sub["timesteps"].to_numpy()
        y = sub["mean"].to_numpy()
        yerr = sub["std"].to_numpy()

        label = "LLM basic prompt" if "basic" in variant else "LLM structured prompt"
        plt.plot(x, y, marker="o", label=label)

        if not np.all(np.isnan(yerr)):
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Cumulative training timesteps (residual)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    basic = load_variant("llm_basic")
    structured = load_variant("llm_structured")
    df = pd.concat([basic, structured], ignore_index=True)

    # Reward plot
    agg_reward = agg_across_seeds(df, value_col="mean_reward")
    plot_two_curves(
        agg_reward,
        title="Pendulum: residual learning with LLM priors (basic vs structured prompt)",
        ylabel="Total reward (mean over seeds & episodes)",
        outpath="logs/pendulum_llm_basic_vs_structured_reward.png",
    )

    # Upright fraction plot (optional)
    agg_upright = agg_across_seeds(df, value_col="mean_upright_fraction")
    plot_two_curves(
        agg_upright,
        title="Pendulum: stability with LLM priors (basic vs structured prompt)",
        ylabel="Upright fraction (mean over seeds & episodes)",
        outpath="logs/pendulum_llm_basic_vs_structured_upright.png",
    )


if __name__ == "__main__":
    main()
